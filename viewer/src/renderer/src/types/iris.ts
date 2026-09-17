export interface HistogramValue {
  count: number
  sum: number
}
export interface MetricsSnapshot {
  counters: Record<string, number>
  gauges: Record<string, number>
  histograms: Record<string, HistogramValue>
}
export interface PoseFrame {
  sourceSequence?: number
  sequence?: number
  joints?: number[][]
  joints3d?: number[][]
  joints_3d?: number[][]
  valid?: boolean[]
  jointValid?: boolean[]
}
