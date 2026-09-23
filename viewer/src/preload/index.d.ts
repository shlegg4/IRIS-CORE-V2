import { ElectronAPI } from '@electron-toolkit/preload'
import type { CreateCameraRequest, DiscoveredCamera, UpdateCameraRequest } from '../renderer/src/types/iris'

declare global {
  interface Window {
    electron: ElectronAPI
    api: {
      request(path: string, method?: string, body?: unknown): Promise<unknown>
      getStatus(): Promise<unknown>
      getMetrics(prefix?: string): Promise<unknown>
      listCameras(): Promise<unknown>
      discoverCameras(): Promise<DiscoveredCamera[]>
      startPipeline(): Promise<unknown>
      stopPipeline(): Promise<unknown>
      startRecording(body: unknown): Promise<unknown>
      stopRecording(): Promise<unknown>
      configureCamera(id: number, body: UpdateCameraRequest): Promise<unknown>
      addCamera(body: CreateCameraRequest): Promise<unknown>
      removeCamera(id: number): Promise<unknown>
      configurePose(body: unknown): Promise<unknown>
      configurePreview(body: unknown): Promise<unknown>
      configureSynchronizer(body: unknown): Promise<unknown>
      getCalibration(): Promise<unknown>
      startCalibration(body?: unknown): Promise<unknown>
      cancelCalibration(): Promise<unknown>
      clearCalibration(): Promise<unknown>
      shutdownRuntime(): Promise<unknown>
      stop(): Promise<void>
      onLog(callback: (log: string) => void): () => void
      onMetrics(callback: (metrics: unknown) => void): () => void
      onPoseFrame(callback: (frame: unknown) => void): () => void
      onStatus(callback: (status: unknown) => void): () => void
    }
  }
}
