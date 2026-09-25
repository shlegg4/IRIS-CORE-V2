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
export interface RuntimeLogEntry {
  id: number
  timestamp: string
  source: 'stdout' | 'stderr' | 'viewer'
  level: 'info' | 'error'
  message: string
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
  source?: string
  cameras: CameraExtrinsic[]
}
export type CameraRotation = 'none' | 'cw90' | '180' | 'ccw90'
export interface CameraStatus {
  camera_id: number
  connected?: boolean
  state?: string
  last_error?: string | null
  frames_received?: number
  frames_dropped?: number
  source_errors?: number
  last_frame_timestamp?: string
  reconnect_count?: number | null
  device_index?: number | null
  device_symbolic_link?: string
  width: number
  height: number
  fps: number
  frame_rate?: { numerator: number; denominator: number; value?: number }
  format?: string
  cuda_device?: number
  sample_queue_capacity?: number
  frame_pool_capacity?: number
  overflow?: string
  rotation?: CameraRotation
  allow_format_fallback?: boolean
  reconnect: boolean
}
export interface DiscoveredCamera {
  name: string
  device_index: number
  device_symbolic_link: string
}
export interface CameraCaptureSettings {
  device_index?: number | null
  device_symbolic_link?: string
  width: number
  height: number
  frame_rate: number
  format: string
  cuda_device: number
  sample_queue_capacity: number
  frame_pool_capacity: number
  overflow: string
  rotation: CameraRotation
  allow_format_fallback: boolean
  reconnect: boolean
}
export interface CreateCameraRequest extends CameraCaptureSettings {
  camera_id: number
}
export type UpdateCameraRequest = Partial<CameraCaptureSettings>
export interface RuntimeStatus {
  api_connected?: boolean
  metrics?: MetricsSnapshot
  state?: string
  input_mode?: 'live' | 'video' | string
  video_inputs?: Array<{ camera_id: number; path: string; rotation?: CameraRotation }>
  video_decode_status?: VideoDecodeStatus[]
  video_cuda_device?: number
  video_frame_pool_capacity?: number
  video_realtime?: boolean
  video_loop?: boolean
  recording?: boolean
  recording_path?: string
  shared_memory_enabled?: boolean
  shared_memory_destination?: string
  cameras?: CameraStatus[]
  processed_packets?: number
  sync_tolerance_ms?: number
  sync_queue_capacity?: number
  incomplete_batch_policy?: string
  pose_backend?: string
  pose_model_path?: string
  pose_engine_path?: string
  last_error?: string
  calibration_tool?: {
    state: string
    message: string
    source_sequence: number
  }
  preview?: {
    bind_address?: string
    port?: number
    enabled?: boolean
    published_packets?: number
    dropped_packets?: number
    connected_clients?: number
    event_clients?: number
    h264_clients?: number
    mjpeg_clients?: number
    last_error?: string
  }
  calibration?: CalibrationSnapshot | null
}
export interface VideoDecodeStatus {
  camera_id: number
  codec: string
  backend: 'NVDEC' | 'SOFTWARE' | string
  detail: string
}
