import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

// Custom APIs for renderer
const api = {
  sendCommand: (command: string) => ipcRenderer.invoke('iris:command', command),
  stop: () => ipcRenderer.invoke('iris:stop'),
  onLog: (callback: (log: string) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, log: string): void => callback(log)
    ipcRenderer.on('iris:log', listener)
    return () => ipcRenderer.removeListener('iris:log', listener)
  },
  onMetrics: (callback: (metrics: unknown) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, metrics: unknown): void =>
      callback(metrics)
    ipcRenderer.on('iris:metrics', listener)
    return () => ipcRenderer.removeListener('iris:metrics', listener)
  },
  onPoseFrame: (callback: (frame: unknown) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, frame: unknown): void => callback(frame)
    ipcRenderer.on('iris:pose', listener)
    return () => ipcRenderer.removeListener('iris:pose', listener)
  },
  onStatus: (callback: (status: unknown) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, status: unknown): void => callback(status)
    ipcRenderer.on('iris:status', listener)
    return () => ipcRenderer.removeListener('iris:status', listener)
  }
}

// Use `contextBridge` APIs to expose Electron APIs to
// renderer only if context isolation is enabled, otherwise
// just add to the DOM global.
if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', electronAPI)
    contextBridge.exposeInMainWorld('api', api)
  } catch (error) {
    console.error(error)
  }
} else {
  // @ts-ignore (define in dts)
  window.electron = electronAPI
  // @ts-ignore (define in dts)
  window.api = api
}
