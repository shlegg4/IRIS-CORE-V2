import { app, shell, BrowserWindow, ipcMain } from 'electron'
import { spawn, type ChildProcessWithoutNullStreams } from 'child_process'
import { dirname, join } from 'path'
import { existsSync, readFile } from 'fs'
import WebSocket from 'ws'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'

let irisProcess: ChildProcessWithoutNullStreams | undefined
let metricsTimer: NodeJS.Timeout | undefined
let previewSocket: WebSocket | undefined
let reconnectTimer: NodeJS.Timeout | undefined

function send(window: BrowserWindow, channel: string, payload: unknown): void {
  if (!window.isDestroyed() && !window.webContents.isDestroyed()) {
    window.webContents.send(channel, payload)
  }
}

function startMetricsBridge(window: BrowserWindow, directory: string): void {
  if (metricsTimer) clearInterval(metricsTimer)
  const path = join(directory, 'iris_metrics.json')
  metricsTimer = setInterval(() => {
    readFile(path, 'utf8', (error, contents) => {
      if (error) return
      try {
        send(window, 'iris:metrics', JSON.parse(contents))
      } catch {
        // The exporter replaces this file periodically; ignore partially written snapshots.
      }
    })
  }, 500)
}

function connectPreviewEvents(window: BrowserWindow): void {
  previewSocket?.close()
  previewSocket = new WebSocket('ws://127.0.0.1:8080/api/events')
  previewSocket.on('message', (data) => {
    try {
      const envelope = JSON.parse(data.toString()) as { type?: string; data?: unknown }
      if (envelope.type === 'pose') send(window, 'iris:pose', envelope.data)
      else if (envelope.type === 'status') send(window, 'iris:status', envelope.data)
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
    irisProcess = spawn(executable, [], {
      cwd: dirname(executable),
      stdio: ['pipe', 'pipe', 'pipe']
    })
    const forward = (_channel: string, chunk: Buffer): void => {
      for (const line of chunk.toString().split(/\r?\n/).filter(Boolean)) {
        try {
          const message = JSON.parse(line) as { type?: string; [key: string]: unknown }
          if (message.type === 'metrics') window.webContents.send('iris:metrics', message)
          else if (message.type === 'pose') window.webContents.send('iris:pose', message)
          else window.webContents.send('iris:log', line)
        } catch {
          window.webContents.send('iris:log', line)
        }
      }
    }
    irisProcess.stdout.on('data', (chunk) => forward('iris:log', chunk))
    irisProcess.stderr.on('data', (chunk) => forward('iris:log', chunk))
    irisProcess.on('error', (error) =>
      window.webContents.send('iris:status', { state: 'error', message: error.message })
    )
    irisProcess.on('close', (code) => {
      window.webContents.send('iris:status', { state: 'stopped', code })
      irisProcess = undefined
    })
    window.webContents.send('iris:status', { state: 'running', executable })
    startMetricsBridge(window, dirname(executable))
    irisProcess.stdin.write('preview enable 8080\n')
    connectPreviewEvents(window)
  } catch (error) {
    window.webContents.send('iris:status', { state: 'error', message: String(error) })
  }
}

function stopIrisRuntime(): void {
  if (metricsTimer) clearInterval(metricsTimer)
  if (reconnectTimer) clearTimeout(reconnectTimer)
  metricsTimer = undefined
  reconnectTimer = undefined
  previewSocket?.close()
  previewSocket = undefined
  if (!irisProcess) return
  irisProcess.kill()
  irisProcess = undefined
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
  ipcMain.handle('iris:command', (_event, command: string) => {
    if (!irisProcess?.stdin.writable) throw new Error('IRIS runtime is not running')
    irisProcess.stdin.write(`${command}\n`)
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
