<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { MetricsSnapshot, VideoDecodeStatus } from '../types/iris'

const props = defineProps<{
  snapshot?: MetricsSnapshot | null
  inputMode?: string
  videoDecodeStatus?: VideoDecodeStatus[]
  activeCameraCount?: number
  cameraCount?: number
}>()
const average = (name: string): number | null => {
  const histogram = props.snapshot?.histograms?.[name]
  return histogram?.count ? histogram.sum / histogram.count : null
}
const metricValue = (names: string[]): number | null => {
  for (const name of names) {
    const value = props.snapshot?.gauges?.[name] ?? props.snapshot?.counters?.[name]
    if (typeof value === 'number' && Number.isFinite(value)) return value
  }
  return null
}
const videoCameraDecodes = computed(() => {
  const gauges = props.snapshot?.gauges ?? {}
  const decodeStatus = new Map((props.videoDecodeStatus ?? []).map((status) => [status.camera_id, status]))
  return Object.entries(gauges)
    .flatMap(([name, value]) => {
      const match = name.match(/^iris_video_camera_(\d+)_last_decode_ms$/)
      if (!match) return []
      const cameraId = Number(match[1])
      const sourceRate = gauges[`iris_video_camera_${cameraId}_source_frame_rate_fps`]
      const nvdec = gauges[`iris_video_camera_${cameraId}_nvdec_active`] === 1
      const gpuConversion = gauges[`iris_video_camera_${cameraId}_gpu_conversion_active`] === 1
      const status = decodeStatus.get(cameraId)
      return [{ cameraId, value, sourceRate: sourceRate ?? null, nvdec, gpuConversion, codec: status?.codec, decodeDetail: status?.detail }]
    })
    .sort((a, b) => a.cameraId - b.cameraId)
})
const stageTimings = computed(() => {
  if (props.inputMode === 'video') {
    const stages = [
      { name: 'VIDEO READ', description: 'Decode and convert every feed in the batch', value: metricValue(['iris_video_last_batch_decode_ms']), tone: 'cyan' },
      { name: 'GPU UPLOAD', description: 'Upload decoded frames and wait for frame buffers', value: (metricValue(['iris_video_last_batch_upload_ms']) ?? 0) + (metricValue(['iris_video_last_batch_pool_wait_ms']) ?? 0), tone: 'green' },
      { name: 'PIPELINE SEND', description: 'Wait to submit the batch downstream', value: metricValue(['iris_video_last_batch_submit_wait_ms']), tone: 'orange' }
    ]
    const maximum = Math.max(...stages.map((stage) => stage.value ?? 0), 1)
    return stages.map((stage) => ({ ...stage, width: `${stage.value === null ? 0 : Math.max(6, (stage.value / maximum) * 100)}%` }))
  }
  const stages = [
    { name: 'CAPTURE', description: 'Latest acquire/decode completion', value: metricValue(['iris_capture_last_capture_to_emit_ms']) ?? average('iris_capture_capture_to_emit_ms'), tone: 'cyan' },
    { name: 'POSE', description: 'Run pose estimation on the packet', value: metricValue(['iris_pose_last_process_ms']) ?? average('iris_pose_process_ms'), tone: 'green' },
    { name: 'OUTPUT', description: 'Deliver the processed packet downstream', value: average('iris_channel_pose_to_output_residence_ms'), tone: 'orange' }
  ]
  const maximum = Math.max(...stages.map((stage) => stage.value ?? 0), 1)
  return stages.map((stage) => ({ ...stage, width: `${stage.value === null ? 0 : Math.max(6, (stage.value / maximum) * 100)}%` }))
})
const captureRate = computed(() => {
  if (props.inputMode === 'video')
    return metricValue(['iris_video_source_frame_rate_fps', 'iris_video_batch_rate_fps'])
  const interval = average('iris_capture_interframe_interval_ms')
  return interval && interval > 0 ? 1000 / interval : null
})
const poseLatency = computed(() =>
  metricValue(['iris_pose_last_capture_to_result_ms']) ?? average('iris_pose_capture_to_result_ms')
)
const droppedFrames = computed(() => {
  const counters = props.snapshot?.counters
  if (!counters) return null
  if (props.inputMode === 'video') {
    const videoDrops = Object.entries(counters)
      .filter(([name]) => name.startsWith('iris_video_') && /dropped.*total|total.*dropped/i.test(name))
    return videoDrops.length ? videoDrops.reduce((sum, [, value]) => sum + value, 0) : null
  }
  return (counters.iris_channel_capture_samples_dropped_total ?? 0) +
    (counters.iris_channel_capture_to_pose_dropped_total ?? 0)
})
type CaptureRateSample = { timestamp: number; value: number }
const captureRateStorageKey = 'iris-viewer-capture-rate-v1'
const captureRateWindowMs = 60_000
const maxCaptureRateSamples = 3_600
const captureRateSmoothingMs = 1_000
function restoreCaptureRateHistory(): CaptureRateSample[] {
  try {
    const stored: unknown = JSON.parse(localStorage.getItem(captureRateStorageKey) ?? '[]')
    if (!Array.isArray(stored)) return []
    const now = Date.now()
    return stored
      .filter((sample): sample is CaptureRateSample => {
        if (!sample || typeof sample !== 'object') return false
        const { timestamp, value } = sample as CaptureRateSample
        return Number.isFinite(timestamp) && Number.isFinite(value) && timestamp <= now && now - timestamp <= captureRateWindowMs
      })
      .slice(-maxCaptureRateSamples)
  } catch {
    return []
  }
}
const captureRateHistory = ref<CaptureRateSample[]>(restoreCaptureRateHistory())
const chartTime = ref(Date.now())
let historyTimer: number | undefined
let persistenceTimer: number | undefined
const initialCaptureHistogram = props.snapshot?.histograms?.iris_capture_interframe_interval_ms
let previousCaptureHistogram: { count: number; sum: number } | null = initialCaptureHistogram
  ? { count: initialCaptureHistogram.count, sum: initialCaptureHistogram.sum }
  : null
watch(() => props.snapshot, (snapshot) => {
  if (!snapshot) return
  let value = captureRate.value
  if (props.inputMode !== 'video') {
    const histogram = snapshot.histograms?.iris_capture_interframe_interval_ms
    if (histogram) {
      const previous = previousCaptureHistogram
      if (previous && histogram.count > previous.count && histogram.sum > previous.sum) {
        const intervalCount = histogram.count - previous.count
        const intervalSum = histogram.sum - previous.sum
        value = intervalSum > 0 ? 1000 * intervalCount / intervalSum : value
      }
      previousCaptureHistogram = { count: histogram.count, sum: histogram.sum }
    }
  }
  const timestamp = Date.now()
  const recentHistory = captureRateHistory.value
    .filter((sample) => timestamp - sample.timestamp <= captureRateWindowMs)
  if (value !== null && Number.isFinite(value)) {
    const previous = recentHistory.at(-1)
    const elapsed = previous ? timestamp - previous.timestamp : 0
    const smoothedValue = previous && elapsed > 0 && elapsed <= 5_000
      ? previous.value + (1 - Math.exp(-elapsed / captureRateSmoothingMs)) * (value - previous.value)
      : value
    recentHistory.push({ timestamp, value: smoothedValue })
  }
  captureRateHistory.value = recentHistory.slice(-maxCaptureRateSamples)
}, { deep: true })
watch(captureRateHistory, () => {
  if (persistenceTimer !== undefined) return
  persistenceTimer = window.setTimeout(() => {
    persistenceTimer = undefined
    try {
      localStorage.setItem(captureRateStorageKey, JSON.stringify(captureRateHistory.value))
    } catch {
      // Keep the live chart usable when storage is unavailable.
    }
  }, 1_000)
}, { deep: true })
onMounted(() => {
  historyTimer = window.setInterval(() => {
    chartTime.value = Date.now()
    const currentHistory = captureRateHistory.value
    const recentHistory = currentHistory.filter((sample) => chartTime.value - sample.timestamp <= captureRateWindowMs)
    if (recentHistory.length !== currentHistory.length)
      captureRateHistory.value = recentHistory
  }, 500)
})
onBeforeUnmount(() => {
  if (historyTimer !== undefined) window.clearInterval(historyTimer)
  if (persistenceTimer !== undefined) {
    window.clearTimeout(persistenceTimer)
    try {
      localStorage.setItem(captureRateStorageKey, JSON.stringify(captureRateHistory.value))
    } catch {
      // Keep the live chart usable when storage is unavailable.
    }
  }
})
const captureRatePath = computed(() => {
  const points = captureRateHistory.value
    .filter((sample) => chartTime.value - sample.timestamp <= captureRateWindowMs)
    .map((sample) => ({
      x: ((sample.timestamp - (chartTime.value - captureRateWindowMs)) / captureRateWindowMs) * 100,
      y: 34 - Math.max(0, Math.min(30, sample.value / 3))
    }))
  if (points.length < 2) return ''
  let path = `M ${points[0].x},${points[0].y}`
  for (let index = 0; index < points.length - 1; index += 1) {
    const previous = points[Math.max(0, index - 1)]
    const current = points[index]
    const next = points[index + 1]
    const following = points[Math.min(points.length - 1, index + 2)]
    const segmentMinY = Math.min(current.y, next.y)
    const segmentMaxY = Math.max(current.y, next.y)
    const controlY1 = Math.max(segmentMinY, Math.min(segmentMaxY, current.y + (next.y - previous.y) / 6))
    const controlY2 = Math.max(segmentMinY, Math.min(segmentMaxY, next.y - (following.y - current.y) / 6))
    path += ` C ${current.x + (next.x - previous.x) / 6},${controlY1}`
    path += ` ${next.x - (following.x - current.x) / 6},${controlY2}`
    path += ` ${next.x},${next.y}`
  }
  return path
})
const frameAgeSummary = computed(() => {
  const gauges = props.snapshot?.gauges ?? {}
  const ages = Object.entries(gauges)
    .filter(([name, value]) => /^iris_capture_camera_\d+_last_frame_age_ms$/.test(name) && Number.isFinite(value))
    .map(([, value]) => value)
  if (!ages.length) {
    const current = metricValue(['iris_capture_last_frame_age_ms'])
    return { oldest: current, average: current }
  }
  return {
    oldest: Math.max(...ages),
    average: ages.reduce((sum, value) => sum + value, 0) / ages.length
  }
})
const queueSummary = computed(() => {
  const value = (name: string): number | null => metricValue([name])
  if (props.inputMode === 'video') return [
    { label: 'Source rate', value: metricValue(['iris_video_source_frame_rate_fps']), unit: 'FPS' },
    { label: 'Ingest rate', value: metricValue(['iris_video_batch_rate_fps']), unit: 'BATCH/S' },
    { label: 'Last decode', value: metricValue(['iris_video_last_batch_decode_ms']), unit: 'MS' },
    { label: 'Frame pool wait', value: metricValue(['iris_video_last_batch_pool_wait_ms']), unit: 'MS' }
  ]
  return [
    { label: 'Capture queue', value: value('iris_channel_capture_samples_depth'), unit: 'FRAMES' },
    { label: 'Pose queue', value: value('iris_channel_capture_to_pose_depth'), unit: 'FRAMES' },
    { label: 'Output queue', value: value('iris_channel_pose_to_output_depth'), unit: 'FRAMES' },
    { label: 'Oldest frame age', value: frameAgeSummary.value.oldest, unit: 'MS' },
    { label: 'Average frame age', value: frameAgeSummary.value.average, unit: 'MS' }
  ]
})

const gaugeFilter = ref('')
const gaugeHistory = ref<Record<string, number[]>>({})
const historyLimit = 120
watch(
  () => props.snapshot?.gauges,
  (gauges) => {
    if (!gauges) return
    const next = { ...gaugeHistory.value }
    for (const [name, value] of Object.entries(gauges)) {
      if (!Number.isFinite(value)) continue
      next[name] = [...(next[name] ?? []), value].slice(-historyLimit)
    }
    gaugeHistory.value = next
  },
  { deep: true }
)
const gaugeCharts = computed(() => {
  const needle = gaugeFilter.value.trim().toLowerCase()
  return Object.entries(gaugeHistory.value)
    .filter(([name]) => !needle || name.toLowerCase().includes(needle))
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, values]) => ({ name, values, current: values.at(-1) ?? 0 }))
})
const chartPoints = (values: number[]): string => {
  if (!values.length) return ''
  const low = Math.min(...values)
  const high = Math.max(...values)
  const range = high - low || 1
  return values
    .map(
      (value, index) =>
        `${(index / Math.max(1, values.length - 1)) * 100},${50 - ((value - low) / range) * 46}`
    )
    .join(' ')
}

type MetricRow = {
  name: string
  kind: 'Counter' | 'Gauge' | 'Histogram'
  value: string
  detail: string
}
const format = (value: number): string =>
  Number.isInteger(value)
    ? value.toLocaleString()
    : value.toLocaleString(undefined, { maximumFractionDigits: 3 })
const allMetrics = computed<MetricRow[]>(() => {
  const snapshot = props.snapshot
  if (!snapshot) return []
  const rows: MetricRow[] = []
  for (const [name, value] of Object.entries(snapshot.counters ?? {}))
    rows.push({ name, kind: 'Counter', value: format(value), detail: 'Cumulative total' })
  for (const [name, value] of Object.entries(snapshot.gauges ?? {}))
    rows.push({ name, kind: 'Gauge', value: format(value), detail: 'Current value' })
  for (const [name, value] of Object.entries(snapshot.histograms ?? {})) {
    const mean = value.count ? value.sum / value.count : 0
    const buckets = value.bounds?.length ? ` · ${value.bounds.length} buckets` : ''
    rows.push({
      name,
      kind: 'Histogram',
      value: `${format(value.count)} samples`,
      detail: `Mean ${format(mean)} · Sum ${format(value.sum)}${buckets}`
    })
  }
  return rows.sort((left, right) => left.name.localeCompare(right.name))
})
</script>

<template>
  <div class="metrics-page">
    <header class="metrics-hero">
      <div>
        <h1>System Metrics</h1>
        <p>Live performance and pipeline health</p>
      </div>
      <div class="metrics-range"><span>{{ snapshot ? 'Live' : 'Waiting for IRIS' }}</span><span>Last 60 seconds</span></div>
    </header>
    <section class="metrics-summary" aria-label="Current performance">
      <article class="summary-stat">
        <strong>{{ captureRate?.toFixed(1) ?? '—' }} <small>FPS</small></strong>
        <span>Capture Rate</span>
      </article>
      <article class="summary-stat">
        <strong>{{ poseLatency?.toFixed(1) ?? '—' }} <small>MS</small></strong>
        <span>Pose Latency (avg)</span>
      </article>
      <article class="summary-stat">
        <strong>{{ droppedFrames?.toLocaleString() ?? '—' }} <small>FRAMES</small></strong>
        <span>Dropped Frames</span>
      </article>
      <article class="summary-stat">
        <strong>{{ activeCameraCount ?? 0 }} / {{ cameraCount ?? 0 }}</strong>
        <span>Active Cameras</span>
      </article>
    </section>
    <section class="capture-chart" aria-labelledby="capture-rate-title">
      <header class="section-heading">
        <div><h2 id="capture-rate-title">Capture Rate</h2><span>FPS</span></div>
        <span>Target {{ captureRate ? `${captureRate.toFixed(0)} FPS` : '—' }}</span>
      </header>
      <div class="chart-stage">
        <div class="chart-y-labels" aria-hidden="true"><span>90</span><span>60</span><span>30</span><span>0</span></div>
        <svg viewBox="0 0 100 40" preserveAspectRatio="none" role="img" aria-label="Capture rate during the last 60 seconds">
          <line x1="0" y1="4" x2="100" y2="4" />
          <line x1="0" y1="14" x2="100" y2="14" />
          <line x1="0" y1="24" x2="100" y2="24" />
          <line x1="0" y1="34" x2="100" y2="34" />
          <path v-if="captureRateHistory.length > 1" :d="captureRatePath" />
        </svg>
      </div>
      <div class="chart-x-labels" aria-hidden="true"><span>60s</span><span>45s</span><span>30s</span><span>15s</span><span>0s</span></div>
      <p v-if="!captureRateHistory.length" class="chart-empty">Waiting for capture metrics.</p>
    </section>
    <div class="metrics-detail-grid">
      <section class="timing-panel" aria-labelledby="pipeline-timing-title">
        <header class="section-heading">
          <div><h2 id="pipeline-timing-title">Pipeline Timing</h2><span>{{ inputMode === 'video' ? 'Latest batch' : 'Average per frame' }}</span></div>
          <span>MS</span>
        </header>
        <div class="timing-list">
          <div v-for="stage in stageTimings" :key="stage.name" class="timing-row">
            <span>{{ stage.name === 'POSE' ? 'Pose Estimation' : stage.name === 'CAPTURE' ? 'Capture' : stage.name === 'OUTPUT' ? 'Output' : stage.name }}</span>
            <i class="timing-meter"><b :class="stage.tone" :style="{ width: stage.width }"></b></i>
            <strong>{{ stage.value?.toFixed(1) ?? '—' }} ms</strong>
          </div>
        </div>
      </section>
      <section class="queues-panel" aria-labelledby="queues-title">
        <header class="section-heading">
          <div><h2 id="queues-title">{{ inputMode === 'video' ? 'Video Input' : 'Queues & Frame Age' }}</h2></div>
          <span>{{ inputMode === 'video' ? 'CURRENT' : 'CURRENT DEPTH' }}</span>
        </header>
        <div class="queue-list">
          <div v-for="item in queueSummary" :key="item.label" class="queue-row">
            <span>{{ item.label }}</span><strong>{{ item.value?.toFixed(item.unit === 'MS' ? 1 : 0) ?? '—' }} <small>{{ item.unit }}</small></strong>
          </div>
          <div v-if="inputMode === 'video'" class="queue-row">
            <span>Active video feeds</span><strong>{{ videoDecodeStatus?.length ?? videoCameraDecodes.length }}</strong>
          </div>
        </div>
      </section>
    </div>
    <details class="metrics-more">
      <summary>Detailed metrics <span>{{ gaugeCharts.length }} gauges · {{ allMetrics.length }} exported series</span></summary>
      <section v-if="inputMode === 'video'" class="video-ingestion">
        <div class="details-heading">Video Feed Detail <span>{{ Number(snapshot?.counters?.iris_video_batches_emitted_total ?? 0).toLocaleString() }} batches · {{ Number(snapshot?.counters?.iris_video_frames_emitted_total ?? 0).toLocaleString() }} frames emitted</span></div>
        <div v-if="videoCameraDecodes.length" class="video-camera-metrics">
          <article v-for="camera in videoCameraDecodes" :key="camera.cameraId" class="video-camera-metric">
            <strong>Camera {{ String(camera.cameraId + 1).padStart(2, '0') }}</strong>
            <span>{{ camera.value.toFixed(1) }} ms last decode</span>
            <small>{{ camera.nvdec ? 'NVDEC' : 'Software decode' }} · {{ camera.gpuConversion ? 'GPU color conversion' : 'CPU color conversion' }}</small>
            <small v-if="camera.codec">{{ camera.codec.toUpperCase() }} · {{ camera.decodeDetail }}</small>
            <small>{{ camera.sourceRate?.toFixed(1) ?? '—' }} FPS source rate · Histogram mean {{ average(`iris_video_camera_${camera.cameraId}_decode_ms`)?.toFixed(1) ?? '—' }} ms</small>
          </article>
        </div>
        <p v-else class="empty-metrics">Waiting for video feed decode metrics.</p>
      </section>
      <section class="gauge-section">
        <header class="details-heading">Gauge History <span>{{ gaugeCharts.length }} live series · last {{ historyLimit / 2 }} seconds</span></header>
        <label class="gauge-filter"><span>Filter gauges</span><input v-model="gaugeFilter" type="search" placeholder="e.g. queue, camera, pose" /></label>
        <div v-if="gaugeCharts.length" class="gauge-grid">
          <article v-for="gauge in gaugeCharts" :key="gauge.name" class="gauge-chart">
            <header><code>{{ gauge.name }}</code><strong>{{ format(gauge.current) }}</strong></header>
            <svg viewBox="0 0 100 54" preserveAspectRatio="none" role="img" :aria-label="`${gauge.name} history`">
              <line x1="0" y1="50" x2="100" y2="50" /><polyline :points="chartPoints(gauge.values)" />
            </svg>
            <footer><span>{{ format(Math.min(...gauge.values)) }}</span><span>{{ format(Math.max(...gauge.values)) }}</span></footer>
          </article>
        </div>
        <p v-else class="empty-metrics">No gauges match this filter.</p>
      </section>
      <section class="all-metrics">
        <header class="details-heading">All Exported Metrics <span>{{ allMetrics.length }} live series</span></header>
        <div v-if="allMetrics.length" class="metric-table" role="table">
          <div class="metric-row metric-header" role="row"><span>Metric</span><span>Type</span><span>Value</span><span>Details</span></div>
          <div v-for="metric in allMetrics" :key="metric.name" class="metric-row" role="row">
            <code>{{ metric.name }}</code><span>{{ metric.kind }}</span><strong>{{ metric.value }}</strong><span>{{ metric.detail }}</span>
          </div>
        </div>
        <p v-else class="empty-metrics">Waiting for the runtime to publish metrics.</p>
      </section>
    </details>
  </div>
</template>

<style scoped>
.metrics-page {
  width: 100%;
  min-width: 0;
  min-height: 0;
  flex: 1 1 auto;
  padding: 28px 30px 36px;
  overflow: hidden auto;
  color: #d5e0e2;
  scrollbar-width: thin;
  scrollbar-color: #2a3b3f transparent;
}
.metrics-hero {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 20px;
  padding-bottom: 24px;
  border-bottom: 1px solid #1e292d;
}
.eyebrow,
.card-title {
  color: #70e3e0;
  font-size: 9px;
  letter-spacing: 0.14em;
}
h2 {
  margin: 8px 0;
  font-size: clamp(19px, 3vw, 24px);
  font-weight: 400;
}
p {
  margin: 0;
  color: #718087;
}
.healthy {
  flex: 0 0 auto;
  padding: 5px 8px;
  border: 1px solid #285a4d;
  border-radius: 3px;
  color: #79dbb0;
  font-size: 9px;
}
.metrics-cards {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 22px 0;
}
.metric-card,
.all-metrics,
.gauge-section {
  min-width: 0;
  padding: 18px;
  border: 1px solid #1e292d;
  background: #10181b;
}
.gauge-section {
  margin: 22px 0;
}
.pipeline-timing {
  margin: 22px 0;
  padding: 18px;
  border: 1px solid #1e292d;
  background: #10181b;
}
.video-ingestion { margin: 22px 0; padding: 18px; border: 1px solid #1e292d; background: #10181b; }
.video-ingestion > .card-title { display: flex; justify-content: space-between; gap: 12px; }
.video-ingestion > .card-title span { color: #718087; font-size: 8px; }
.video-camera-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; margin-top: 14px; }
.video-camera-metric { display: grid; gap: 6px; min-width: 0; padding: 12px; border: 1px solid #26373b; background: #0c1214; }
.video-camera-metric strong { color: #70e3e0; font-size: 10px; letter-spacing: .08em; }
.video-camera-metric span { color: #d5e0e2; font-size: 12px; }
.video-camera-metric small { color: #718087; font-size: 9px; line-height: 1.45; overflow-wrap: anywhere; }
.pipeline-timing .card-title { display: flex; justify-content: space-between; gap: 12px; }
.pipeline-timing .card-title span { color: #718087; font-size: 8px; }
.pipeline-flow { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 28px; margin-top: 16px; }
.stage-block { position: relative; min-width: 0; padding: 14px; border: 1px solid #26373b; background: #0c1214; }
.stage-heading { display: flex; align-items: center; gap: 9px; }
.stage-index { color: #718087; font-size: 9px; }
.stage-name { color: #d5e0e2; font-size: 10px; letter-spacing: 0.12em; }
.stage-value { display: block; margin-top: 18px; color: #70e3e0; font-size: 25px; font-weight: 400; }
.stage-value small { color: #718087; font-size: 9px; }
.stage-description { display: block; min-height: 28px; margin-top: 5px; color: #718087; font-size: 10px; line-height: 1.4; }
.stage-meter { height: 3px; margin-top: 15px; background: #1b292d; }
.stage-meter i { display: block; height: 100%; }
.stage-arrow { position: absolute; top: 50%; right: -22px; color: #70e3e0; font-size: 18px; }
.gauge-filter {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 16px 0;
  color: #718087;
  font-size: 11px;
}
.gauge-filter input {
  width: min(100%, 320px);
  padding: 7px 9px;
  border: 1px solid #2a3b3f;
  background: #0c1214;
  color: #d5e0e2;
}
.gauge-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}
.gauge-chart {
  min-width: 0;
  padding: 12px;
  border: 1px solid #1e292d;
  background: #0c1214;
}
.gauge-chart header,
.gauge-chart footer {
  display: flex;
  justify-content: space-between;
  gap: 8px;
}
.gauge-chart code {
  overflow: hidden;
  color: #a9b8b9;
  font-size: 9px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.gauge-chart strong {
  color: #70e3e0;
  font-size: 12px;
  font-weight: 400;
}
.gauge-chart svg {
  display: block;
  width: 100%;
  height: 72px;
  margin: 10px 0 4px;
}
.gauge-chart line {
  stroke: #26373b;
  stroke-width: 1;
}
.gauge-chart polyline {
  fill: none;
  stroke: #70e3e0;
  stroke-width: 1.5;
  vector-effect: non-scaling-stroke;
}
.gauge-chart footer {
  color: #718087;
  font-size: 9px;
}
.metric-label,
.metric-detail {
  display: block;
  overflow: hidden;
  color: #718087;
  font-size: 9px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.metric-label {
  letter-spacing: 0.1em;
}
.metric-card strong {
  display: block;
  margin: 16px 0 5px;
  overflow: hidden;
  font-size: clamp(19px, 3vw, 25px);
  font-weight: 400;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.metric-card small {
  color: #70e3e0;
  font-size: 10px;
}
.large-meter {
  height: 4px;
  margin-top: 20px;
  overflow: hidden;
  background: #1c292d;
}
.large-meter i {
  display: block;
  height: 100%;
  background: #70e3e0;
}
.large-meter i.green {
  background: #79dbb0;
}
.large-meter i.orange {
  background: #d9925f;
}
.card-title {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 18px;
}
.card-title span {
  color: #718087;
  font-size: 8px;
  white-space: nowrap;
}
.metric-table {
  overflow-x: auto;
}
.metric-row {
  min-width: 680px;
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 10px 0;
  border-bottom: 1px solid #1e292d;
  color: #a9b8b9;
  font-size: 11px;
}
.metric-row > :nth-child(1) {
  flex: 1.8 1 0;
}
.metric-row > :nth-child(2) {
  flex: 0.6 1 0;
}
.metric-row > :nth-child(3) {
  flex: 0.8 1 0;
}
.metric-row > :nth-child(4) {
  flex: 1.4 1 0;
}
.metric-row code {
  overflow: hidden;
  color: #d5e0e2;
  font-family: ui-monospace, Consolas, monospace;
  font-size: 10px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.metric-row strong {
  color: #70e3e0;
  font-weight: 400;
}
.metric-header {
  color: #718087;
  font-size: 9px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.empty-metrics {
  color: #718087;
  font-size: 12px;
}
@media (max-width: 1000px) {
  .metrics-cards {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .gauge-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .pipeline-flow { gap: 12px; }
  .stage-arrow { right: -13px; }
}
@media (max-width: 760px) {
  .metrics-page {
    padding: 20px 18px 28px;
  }
}
@media (max-width: 480px) {
  .metrics-cards {
    grid-template-columns: 1fr;
  }
  .gauge-grid {
    grid-template-columns: 1fr;
  }
  .pipeline-flow { grid-template-columns: 1fr; gap: 12px; }
  .stage-arrow { top: auto; right: auto; bottom: -18px; left: 50%; transform: rotate(90deg); }
  .gauge-filter {
    align-items: stretch;
    flex-direction: column;
  }
  .metrics-hero p {
    display: none;
  }
}

.metrics-page {
  padding: 22px 28px 28px;
  color: #e6ebee;
  font-size: 13px;
  scrollbar-color: #34434e transparent;
}
.metrics-hero {
  min-height: 58px;
  align-items: center;
  padding: 0 0 14px;
  border-color: #26333e;
}
.metrics-hero h1 {
  margin: 0 0 4px;
  font-size: 20px;
  font-weight: 500;
}
.metrics-hero p {
  color: #9aa8b1;
  font-size: 13px;
}
.metrics-range {
  display: grid;
  gap: 4px;
  color: #9aa8b1;
  font-size: 12px;
  text-align: right;
}
.metrics-range span:first-child {
  color: #36df98;
}
.metrics-summary {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  margin: 14px 0 16px;
  padding: 4px 0 12px;
  border-bottom: 1px solid #26333e;
}
.summary-stat {
  display: grid;
  place-content: center;
  gap: 4px;
  min-width: 0;
  min-height: 62px;
  padding: 0 18px;
  text-align: center;
}
.summary-stat + .summary-stat {
  border-left: 1px solid #34434e;
}
.summary-stat strong {
  overflow: hidden;
  color: #29d8d1;
  font-size: 22px;
  font-weight: 500;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.summary-stat small {
  color: #c1cbd1;
  font-size: 12px;
  font-weight: 400;
}
.summary-stat > span {
  color: #bdc7cd;
  font-size: 12px;
}
.capture-chart,
.timing-panel,
.queues-panel {
  min-width: 0;
  padding: 14px;
  border: 1px solid #293640;
  background: #121c25;
}
.capture-chart {
  position: relative;
  margin-bottom: 12px;
  overflow: hidden;
}
.section-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  color: #9aa8b1;
  font-size: 11px;
}
.section-heading h2 {
  margin: 0 0 3px;
  color: #e6ebee;
  font-size: 13px;
  font-weight: 500;
}
.section-heading div > span,
.section-heading > span {
  color: #9aa8b1;
  font-size: 11px;
}
.chart-stage {
  display: grid;
  grid-template-columns: 28px minmax(0, 1fr);
  align-items: stretch;
  gap: 7px;
  height: clamp(150px, 24vh, 220px);
  margin-top: 8px;
}
.chart-y-labels {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 1px 0 7px;
  color: #9aa8b1;
  font-size: 10px;
  text-align: right;
}
.chart-stage svg {
  display: block;
  width: 100%;
  height: 100%;
  overflow: hidden;
}
.chart-stage line {
  stroke: #34434e;
  stroke-width: .5;
  vector-effect: non-scaling-stroke;
}
.chart-stage path {
  fill: none;
  stroke: #29d8d1;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
  vector-effect: non-scaling-stroke;
}
.chart-x-labels {
  display: flex;
  justify-content: space-between;
  margin: 5px 0 0 35px;
  color: #9aa8b1;
  font-size: 10px;
}
.chart-empty {
  position: absolute;
  right: 35px;
  bottom: 48%;
  left: 55px;
  color: #9aa8b1;
  font-size: 12px;
  text-align: center;
  pointer-events: none;
}
.metrics-detail-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(280px, .9fr);
  gap: 12px;
}
.timing-panel,
.queues-panel {
  min-height: 174px;
}
.timing-list,
.queue-list {
  margin-top: 10px;
}
.timing-row,
.queue-row {
  display: grid;
  grid-template-columns: minmax(100px, .75fr) minmax(80px, 1.5fr) minmax(65px, auto);
  align-items: center;
  gap: 12px;
  min-height: 34px;
  border-top: 1px solid #24313a;
  color: #bdc7cd;
  font-size: 12px;
}
.timing-row strong,
.queue-row strong {
  color: #e6ebee;
  font-size: 12px;
  font-weight: 400;
  text-align: right;
  white-space: nowrap;
}
.timing-meter {
  display: block;
  height: 8px;
  overflow: hidden;
  background: #27343e;
}
.timing-meter b {
  display: block;
  height: 100%;
  background: #29d8d1;
}
.timing-meter b.green {
  background: #36df98;
}
.timing-meter b.orange {
  background: #eba45b;
}
.queue-row {
  grid-template-columns: minmax(0, 1fr) auto;
  min-height: 29px;
}
.queue-row small {
  color: #9aa8b1;
  font-size: 10px;
}
.metrics-more {
  margin-top: 12px;
  border-top: 1px solid #293640;
  border-bottom: 1px solid #293640;
}
.metrics-more > summary {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 2px;
  color: #dce3e7;
  font-size: 12px;
  cursor: pointer;
  list-style-position: inside;
}
.metrics-more > summary span {
  color: #9aa8b1;
  font-size: 11px;
}
.metrics-more > section {
  margin: 0 0 14px;
  padding: 12px;
  border: 1px solid #293640;
  background: #121c25;
}
.details-heading {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  color: #e6ebee;
  font-size: 12px;
}
.details-heading span {
  color: #9aa8b1;
  font-size: 10px;
}
.metrics-more .video-camera-metrics,
.metrics-more .gauge-grid {
  margin-top: 10px;
}
.metrics-more .gauge-section,
.metrics-more .pipeline-timing,
.metrics-more .video-ingestion,
.metrics-more .all-metrics {
  margin: 0;
  padding: 0;
  border: 0;
  background: transparent;
}
.metrics-more .metric-row {
  min-width: 620px;
  font-size: 11px;
}
.metrics-more .metric-row code {
  font-size: 10px;
}
@media (max-width: 900px) {
  .metrics-detail-grid {
    grid-template-columns: 1fr;
  }
}
@media (max-width: 600px) {
  .metrics-page {
    padding: 18px 16px 24px;
  }
  .metrics-summary {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .summary-stat:nth-child(3) {
    border-left: 0;
  }
  .summary-stat:nth-child(n + 3) {
    border-top: 1px solid #34434e;
  }
  .timing-row {
    grid-template-columns: minmax(90px, .75fr) minmax(50px, 1fr) auto;
    gap: 8px;
    font-size: 11px;
  }
}
</style>
