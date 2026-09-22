import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'
import type { CreateCameraRequest, UpdateCameraRequest } from '../renderer/src/types/iris'

// Custom APIs for renderer
const api = {
  request: (path: string, method = 'GET', body?: unknown) => ipcRenderer.invoke('iris:api', path, method, body),
  getStatus: () => ipcRenderer.invoke('iris:api', '/status'),
  getMetrics: (prefix = '') => ipcRenderer.invoke('iris:api', prefix ? `/metrics?prefix=${encodeURIComponent(prefix)}` : '/metrics'),
  listCameras: () => ipcRenderer.invoke('iris:api', '/cameras'),
  startPipeline: () => ipcRenderer.invoke('iris:api', '/pipeline/start', 'POST'),
  stopPipeline: () => ipcRenderer.invoke('iris:api', '/pipeline/stop', 'POST'),
  stopRecording: () => ipcRenderer.invoke('iris:api', '/recording/stop', 'POST'),
  startRecording: (body: unknown) => ipcRenderer.invoke('iris:api', '/recording/start', 'POST', body),
  configureCamera: (id: number, body: UpdateCameraRequest) => ipcRenderer.invoke('iris:api', `/cameras/${id}`, 'PATCH', body),
  addCamera: (body: CreateCameraRequest) => ipcRenderer.invoke('iris:api', '/cameras', 'POST', body),
  removeCamera: (id: number) => ipcRenderer.invoke('iris:api', `/cameras/${id}`, 'DELETE'),
  configurePose: (body: unknown) => ipcRenderer.invoke('iris:api', '/pose', 'PATCH', body),
  configurePreview: (body: unknown) => ipcRenderer.invoke('iris:api', '/outputs/preview', 'PATCH', body),
  configureSynchronizer: (body: unknown) => ipcRenderer.invoke('iris:api', '/synchronizer', 'PATCH', body),
  getCalibration: () => ipcRenderer.invoke('iris:api', '/calibration'),
  startCalibration: (body?: unknown) => ipcRenderer.invoke('iris:api', '/calibration/start', 'POST', body),
  cancelCalibration: () => ipcRenderer.invoke('iris:api', '/calibration/cancel', 'POST'),
  clearCalibration: () => ipcRenderer.invoke('iris:api', '/calibration/clear', 'POST'),
  shutdownRuntime: () => ipcRenderer.invoke('iris:api', '/shutdown', 'POST'),
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
