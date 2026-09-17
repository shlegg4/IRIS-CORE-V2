import { ElectronAPI } from '@electron-toolkit/preload'

declare global {
  interface Window {
    electron: ElectronAPI
    api: {
      sendCommand(command: string): Promise<void>
      stop(): Promise<void>
      onLog(callback: (log: string) => void): () => void
      onMetrics(callback: (metrics: unknown) => void): () => void
      onPoseFrame(callback: (frame: unknown) => void): () => void
      onStatus(callback: (status: unknown) => void): () => void
    }
  }
}
