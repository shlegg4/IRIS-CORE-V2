export interface HistogramValue {
  count: number
  sum: number
  bounds?: number[]
  counts?: number[]
}
export interface MetricsSnapshot {
  counters: Record<string, number>
  gauges: Record<string, number>
  histograms: Record<string, HistogramValue>
}
export interface PosePerson {
  id?: number
  joints?: number[][]
  joints3d?: number[][]
  joints_3d?: number[][]
  valid?: boolean[]
  jointValid?: boolean[]
}
export interface PoseFrame extends PosePerson {
  sourceSequence?: number
  sequence?: number
  people?: PosePerson[]
}
export interface CameraExtrinsic {
  camera_id: number
  R_w2c: number[]
  t_w2c: number[]
}
export interface CalibrationSnapshot {
  revision: number
  cameras: CameraExtrinsic[]
}
export interface CameraStatus {
  camera_id: number
  width: number
  height: number
  fps: number
  reconnect: boolean
}
export interface RuntimeStatus {
  pipeline?: string
  cameras?: CameraStatus[]
  previewDropped?: number
  previewPublished?: number
  preview?: {
    port?: number
    enabled?: boolean
    last_error?: string
    h264?: { enabled?: boolean; connected_clients?: number; published_packets?: number; dropped_packets?: number; last_error?: string; codec?: string; bitrate?: number; max_fps?: number; max_width?: number }
  }
  lastError?: string
  calibration?: CalibrationSnapshot
}
