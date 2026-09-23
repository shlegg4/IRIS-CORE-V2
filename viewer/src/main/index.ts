import { app, shell, BrowserWindow, ipcMain } from 'electron'
import { spawn, type ChildProcessByStdio } from 'child_process'
import type { Readable } from 'stream'
import { dirname, join } from 'path'
import { existsSync } from 'fs'
import WebSocket from 'ws'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'

let irisProcess: ChildProcessByStdio<null, Readable, Readable> | undefined
let apiTimer: NodeJS.Timeout | undefined
let previewSocket: WebSocket | undefined
let reconnectTimer: NodeJS.Timeout | undefined

function send(window: BrowserWindow, channel: string, payload: unknown): void {
  if (!window.isDestroyed() && !window.webContents.isDestroyed()) {
    window.webContents.send(channel, payload)
  }
}

const API_BASE = 'http://127.0.0.1:8090/api/v1'
type ApiResult = { status?: string; message?: string; snapshot?: unknown; [key: string]: unknown }
function normalizeRuntimeStatus(payload: unknown): Record<string, unknown> {
  if (!payload || typeof payload !== 'object') return { state: 'unknown' }
  const value = payload as Record<string, unknown>
  if (typeof value.state === 'string') return value
  if (typeof value.pipeline === 'string') {
    return {
      ...value,
      state: value.pipeline,
      processed_packets: value.processedPackets,
      last_error: value.lastError,
      preview: value.preview
    }
  }
  return value
}
async function apiRequest(path: string, method = 'GET', body?: unknown): Promise<ApiResult> {
  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers: body === undefined ? undefined : { 'content-type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body)
  })
  const data = (await response.json()) as ApiResult
  if (!response.ok) throw new Error(data.message || `IRIS API returned ${response.status}`)
  return data
}
async function waitForApi(): Promise<void> {
  for (let attempt = 0; attempt < 50; attempt++) {
    try { await apiRequest('/status'); return } catch { await new Promise((resolve) => setTimeout(resolve, 100)) }
  }
  throw new Error('IRIS REST API did not become ready')
}
function startApiBridge(window: BrowserWindow): void {
  if (apiTimer) clearInterval(apiTimer)
  const poll = async (): Promise<void> => {
    try {
      const [status, metrics] = await Promise.all([apiRequest('/status'), apiRequest('/metrics')])
      send(window, 'iris:status', normalizeRuntimeStatus(status))
      send(window, 'iris:metrics', metrics)
    } catch (error) { send(window, 'iris:log', `REST polling error: ${String(error)}`) }
  }
  void poll()
  apiTimer = setInterval(() => void poll(), 500)
}

function connectPreviewEvents(window: BrowserWindow): void {
  previewSocket?.close()
  previewSocket = new WebSocket('ws://127.0.0.1:8080/api/events')
  previewSocket.on('message', (data) => {
    try {
      const envelope = JSON.parse(data.toString()) as { type?: string; data?: unknown }
      if (envelope.type === 'pose') send(window, 'iris:pose', envelope.data)
    } catch (error) {
      send(window, 'iris:log', `preview event error: ${String(error)}`)
    }
  })
  previewSocket.on('error', () => undefined)
  previewSocket.on('close', () => {
    previewSocket = undefined
    if (irisProcess && !reconnectTimer) {
      reconnectTimer = setTimeout(() => {
        reconnectTimer = undefined
        connectPreviewEvents(window)
      }, 1000)
    }
  })
}

function runtimePath(): string {
  const executable = process.platform === 'win32' ? 'iris_app.exe' : 'iris_app'
  const candidates = process.env.IRIS_RUNTIME_PATH
    ? [process.env.IRIS_RUNTIME_PATH]
    : [
        join(process.cwd(), '..', 'build', 'default', 'bin', executable),
        join(process.cwd(), '..', 'build', 'hardware', 'bin', executable),
        join(process.cwd(), '..', 'build', 'default', executable),
        join(process.cwd(), '..', 'build', 'bin', executable)
      ]
  const resolved = candidates.find((candidate) => existsSync(candidate))
  if (!resolved) throw new Error(`IRIS executable not found. Checked: ${candidates.join(', ')}`)
  return resolved
}

function startIrisRuntime(window: BrowserWindow): void {
  if (irisProcess) return
  try {
    const executable = runtimePath()
    const child = spawn(executable, ['--api'], {
      cwd: dirname(executable),
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        ...process.env,
        PATH: [dirname(executable), process.env.PATH]
          .filter(Boolean)
          .join(process.platform === 'win32' ? ';' : ':'),
      }
    })
    irisProcess = child
    const forward = (_channel: string, chunk: Buffer): void => {
      for (const line of chunk.toString().split(/\r?\n/).filter(Boolean)) {
        try {
          const message = JSON.parse(line) as { type?: string; [key: string]: unknown }
          if (message.type === 'metrics') window.webContents.send('iris:metrics', message)
          else if (message.type === 'pose')
            window.webContents.send('iris:pose', message.data ?? message)
          else window.webContents.send('iris:log', line)
        } catch {
          window.webContents.send('iris:log', line)
        }
      }
    }
    child.stdout.on('data', (chunk) => forward('iris:log', chunk))
    child.stderr.on('data', (chunk) => forward('iris:log', chunk))
    child.on('error', (error) =>
      send(window, 'iris:status', { state: 'error', message: error.message })
    )
    child.on('close', (code, signal) => {
      // A delayed close event from an older child must not clear a newer
      // runtime started after it.
      if (irisProcess === child) irisProcess = undefined
      send(
        window,
        'iris:log',
        `IRIS runtime exited (code ${code ?? 'none'}, signal ${signal ?? 'none'}).`
      )
      send(window, 'iris:status', { state: 'stopped', code, signal })
    })
    void waitForApi().then(() => {
      send(window, 'iris:status', { state: 'running', executable })
      startApiBridge(window)
      connectPreviewEvents(window)
    }).catch((error) => send(window, 'iris:status', { state: 'error', message: String(error) }))
  } catch (error) {
    window.webContents.send('iris:status', { state: 'error', message: String(error) })
  }
}

function stopIrisRuntime(): void {
  if (apiTimer) clearInterval(apiTimer)
  if (reconnectTimer) clearTimeout(reconnectTimer)
  apiTimer = undefined
  reconnectTimer = undefined
  previewSocket?.close()
  previewSocket = undefined
  if (!irisProcess) return
  const child = irisProcess
  irisProcess = undefined
  void apiRequest('/shutdown').catch(() => undefined).finally(() => {
    setTimeout(() => { if (irisProcess === child) child.kill() }, 1000)
  })
}

function createWindow(): BrowserWindow {
  // Create the browser window.
  const mainWindow = new BrowserWindow({
    width: 900,
    height: 670,
    show: false,
    autoHideMenuBar: true,
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  // HMR for renderer base on electron-vite cli.
  // Load the remote URL for development or the local html file for production.
  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
  return mainWindow
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
  // Set app user model id for windows
  electronApp.setAppUserModelId('com.electron')

  // Default open or close DevTools by F12 in development
  // and ignore CommandOrControl + R in production.
  // see https://github.com/alex8088/electron-toolkit/tree/master/packages/utils
  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  // IPC test
  ipcMain.on('ping', () => console.log('pong'))
  ipcMain.handle('iris:api', async (_event, path: string, method = 'GET', body?: unknown) => apiRequest(path, method, body))
  ipcMain.handle('iris:stop', () => stopIrisRuntime())

  const window = createWindow()
  startIrisRuntime(window)

  app.on('activate', function () {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('before-quit', () => stopIrisRuntime())

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.
