import { app, shell, BrowserWindow, ipcMain, dialog, type OpenDialogOptions } from 'electron'
import { spawn, type ChildProcessByStdio } from 'child_process'
import type { Readable } from 'stream'
import { dirname, join } from 'path'
import { existsSync } from 'fs'
import WebSocket from 'ws'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import type { RuntimeLogEntry } from '../renderer/src/types/iris'
import icon from '../../resources/icon.png?asset'

let irisProcess: ChildProcessByStdio<null, Readable, Readable> | undefined
let apiTimer: NodeJS.Timeout | undefined
let apiPollGeneration = 0
let poseEventsSocket: WebSocket | undefined
let reconnectTimer: NodeJS.Timeout | undefined
const recentLogs: RuntimeLogEntry[] = []
let nextLogId = 1

function send(window: BrowserWindow, channel: string, payload: unknown): void {
  if (!window.isDestroyed() && !window.webContents.isDestroyed()) {
    window.webContents.send(channel, payload)
  }
}
function emitLog(
  window: BrowserWindow,
  source: RuntimeLogEntry['source'],
  message: string,
  level: RuntimeLogEntry['level'] = source === 'stderr' ? 'error' : 'info'
): void {
  const entry: RuntimeLogEntry = {
    id: nextLogId++,
    timestamp: new Date().toISOString(),
    source,
    level,
    message
  }
  recentLogs.push(entry)
  if (recentLogs.length > 500) recentLogs.shift()
  send(window, 'iris:log', entry)
}

const API_BASE = 'http://127.0.0.1:8090/api/v1'
type ApiResult = { status?: string; message?: string; snapshot?: unknown; [key: string]: unknown }
function normalizeRuntimeStatus(payload: unknown): Record<string, unknown> {
  if (!payload || typeof payload !== 'object') return { state: 'unknown' }
  return payload as Record<string, unknown>
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
  stopApiBridge()
  const generation = apiPollGeneration
  const poll = async (): Promise<void> => {
    try {
      const status = await apiRequest('/status')
      if (generation !== apiPollGeneration || !irisProcess) return
      send(window, 'iris:status', normalizeRuntimeStatus(status))
      if (status.metrics) send(window, 'iris:metrics', status.metrics)
    } catch (error) {
      if (generation === apiPollGeneration && irisProcess)
        emitLog(window, 'viewer', `REST polling error: ${String(error)}`, 'error')
    } finally {
      if (generation === apiPollGeneration && irisProcess)
        apiTimer = setTimeout(() => void poll(), 500)
    }
  }
  void poll()
}
function stopApiBridge(): void {
  apiPollGeneration += 1
  if (apiTimer) clearTimeout(apiTimer)
  apiTimer = undefined
}

function connectPoseEvents(window: BrowserWindow): void {
  stopPoseEvents()
  const socket = new WebSocket('ws://127.0.0.1:8080/api/preview/pose-events')
  poseEventsSocket = socket
  socket.on('message', (data) => {
    try {
      const envelope = JSON.parse(data.toString()) as { type?: string; data?: unknown }
      if (envelope.type === 'pose') send(window, 'iris:pose', envelope.data)
    } catch (error) {
      emitLog(window, 'viewer', `preview event error: ${String(error)}`, 'error')
    }
  })
  socket.on('error', () => undefined)
  socket.on('close', () => {
    if (poseEventsSocket !== socket) return
    poseEventsSocket = undefined
    if (irisProcess && !reconnectTimer) {
      reconnectTimer = setTimeout(() => {
        reconnectTimer = undefined
        connectPoseEvents(window)
      }, 1000)
    }
  })
}
function stopPoseEvents(): void {
  if (reconnectTimer) clearTimeout(reconnectTimer)
  reconnectTimer = undefined
  const socket = poseEventsSocket
  poseEventsSocket = undefined
  socket?.terminate()
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
    const forwardLine = (source: 'stdout' | 'stderr', line: string): void => {
      if (source === 'stdout') {
        try {
          const message = JSON.parse(line) as { type?: string; data?: unknown }
          if (message.type === 'metrics') {
            send(window, 'iris:metrics', message.data ?? message)
            return
          }
          if (message.type === 'pose') {
            send(window, 'iris:pose', message.data ?? message)
            return
          }
        } catch { /* Plain runtime output is shown in the terminal. */ }
      }
      emitLog(window, source, line)
    }
    const captureOutput = (stream: Readable, source: 'stdout' | 'stderr'): void => {
      stream.setEncoding('utf8')
      let pending = ''
      stream.on('data', (chunk: string) => {
        pending += chunk
        let newline = pending.indexOf('\n')
        while (newline !== -1) {
          const line = pending.slice(0, newline).replace(/\r$/, '')
          if (line) forwardLine(source, line)
          pending = pending.slice(newline + 1)
          newline = pending.indexOf('\n')
        }
        if (pending.length > 16384) {
          forwardLine(source, pending)
          pending = ''
        }
      })
      stream.on('end', () => {
        if (pending) forwardLine(source, pending.replace(/\r$/, ''))
      })
    }
    captureOutput(child.stdout, 'stdout')
    captureOutput(child.stderr, 'stderr')
    child.on('error', (error) => {
      emitLog(window, 'viewer', `IRIS runtime process error: ${error.message}`, 'error')
      send(window, 'iris:status', { state: 'error', message: error.message })
    })
    child.on('close', (code, signal) => {
      // A delayed close event from an older child must not clear a newer
      // runtime started after it.
      const current = irisProcess === child
      if (current) {
        irisProcess = undefined
        stopApiBridge()
        stopPoseEvents()
      }
      const failed = code !== 0 || signal !== null
      const exitCode = code === null ? 'none' : `${code} (0x${(code >>> 0).toString(16).padStart(8, '0').toUpperCase()})`
      const message = `IRIS runtime exited (code ${exitCode}, signal ${signal ?? 'none'}).`
      emitLog(window, 'viewer', message, failed ? 'error' : 'info')
      if (current || !irisProcess)
        send(window, 'iris:status', { state: failed ? 'error' : 'stopped', message, code, signal })
    })
    void waitForApi().then(() => {
      if (irisProcess !== child) return
      send(window, 'iris:status', { state: 'running', executable })
      startApiBridge(window)
      connectPoseEvents(window)
    }).catch((error) => {
      if (irisProcess !== child) return
      emitLog(window, 'viewer', `IRIS API startup failed: ${String(error)}`, 'error')
      send(window, 'iris:status', { state: 'error', message: String(error) })
    })
  } catch (error) {
    emitLog(window, 'viewer', `IRIS runtime startup failed: ${String(error)}`, 'error')
    send(window, 'iris:status', { state: 'error', message: String(error) })
  }
}

function stopIrisRuntime(): void {
  stopApiBridge()
  stopPoseEvents()
  if (!irisProcess) return
  const child = irisProcess
  irisProcess = undefined
  void apiRequest('/shutdown').catch(() => undefined).finally(() => {
    setTimeout(() => { if (child.exitCode === null && child.signalCode === null) child.kill() }, 1000)
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
  ipcMain.handle('iris:recent-logs', () => recentLogs.slice())
  ipcMain.handle('iris:pick-videos', async (event) => {
    const window = BrowserWindow.fromWebContents(event.sender)
    const options: OpenDialogOptions = {
      title: 'Select synchronized video feeds',
      properties: ['openFile', 'multiSelections'],
      filters: [{ name: 'Video files', extensions: ['mp4', 'mov', 'mkv', 'avi', 'm4v', 'webm'] }]
    }
    const result = window
      ? await dialog.showOpenDialog(window, options)
      : await dialog.showOpenDialog(options)
    return result.canceled ? [] : result.filePaths
  })
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
