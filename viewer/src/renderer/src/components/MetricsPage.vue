<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { MetricsSnapshot } from '../types/iris'

const props = defineProps<{ snapshot?: MetricsSnapshot | null }>()
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
const stageTimings = computed(() => {
  const stages = [
    { name: 'CAPTURE', description: 'Latest acquire/decode completion', value: metricValue(['iris_capture_last_capture_to_emit_ms']) ?? average('iris_capture_capture_to_emit_ms'), tone: 'cyan' },
    { name: 'POSE', description: 'Run pose estimation on the packet', value: metricValue(['iris_pose_last_process_ms']) ?? average('iris_pose_process_ms'), tone: 'green' },
    { name: 'OUTPUT', description: 'Deliver the processed packet downstream', value: average('iris_channel_pose_to_output_residence_ms'), tone: 'orange' }
  ]
  const maximum = Math.max(...stages.map((stage) => stage.value ?? 0), 1)
  return stages.map((stage) => ({ ...stage, width: `${stage.value === null ? 0 : Math.max(6, (stage.value / maximum) * 100)}%` }))
})
const metrics = computed(() => {
  const interval = average('iris_capture_interframe_interval_ms')
  const captureRate = interval && interval > 0 ? 1000 / interval : null
  const captureLatency = metricValue(['iris_capture_last_capture_to_emit_ms']) ?? average('iris_capture_capture_to_emit_ms')
  const decodeSubmit = average('iris_capture_decode_submit_ms')
  const frameAge = metricValue(['iris_capture_last_frame_age_ms'])
  const poseQueue = metricValue(['iris_channel_capture_to_pose_depth'])
  const outputQueue = metricValue(['iris_channel_pose_to_output_depth'])
  const pool = metricValue(['iris_capture_pool_available'])
  const poseLatency =
    metricValue(['iris_pose_last_capture_to_result_ms']) ??
    average('iris_pose_capture_to_result_ms')
  const poseProcess = metricValue(['iris_pose_last_process_ms']) ?? average('iris_pose_process_ms')
  const dropped =
    (metricValue(['iris_channel_capture_samples_dropped_total']) ?? 0) +
    (metricValue(['iris_channel_capture_to_pose_dropped_total']) ?? 0)
  return [
    {
      label: 'POSE LATENCY',
      value: poseLatency?.toFixed(1) ?? '—',
      unit: 'MS',
      detail: `Capture to triangulated pose · ${poseProcess?.toFixed(1) ?? '—'} ms pose processing`,
      width: `${Math.max(4, 100 - Math.min(100, (poseLatency ?? 100) / 5))}%`,
      tone: poseLatency && poseLatency > 150 ? 'orange' : 'cyan'
    },
    {
      label: 'CAPTURE RATE',
      value: captureRate?.toFixed(1) ?? '—',
      unit: 'FPS',
      detail: '4 synchronized sources',
      width: '96%',
      tone: 'green'
    },
    {
      label: 'CAPTURE LATENCY',
      value: captureLatency?.toFixed(1) ?? '—',
      unit: 'MS',
      detail: 'Source timestamp to emitted frame',
      width: `${Math.max(4, 100 - Math.min(100, (captureLatency ?? 100) * 2))}%`,
      tone: 'cyan'
    },
    {
      label: 'DECODE SUBMIT',
      value: decodeSubmit?.toFixed(1) ?? '—',
      unit: 'MS',
      detail: 'Mean GPU decode submission time',
      width: `${Math.max(4, 100 - Math.min(100, (decodeSubmit ?? 100) * 4))}%`,
      tone: 'cyan'
    },
    {
      label: 'FRAME AGE',
      value: frameAge?.toFixed(1) ?? '—',
      unit: 'MS',
      detail: 'Age of the latest captured frame',
      width: `${Math.max(4, 100 - Math.min(100, (frameAge ?? 100) * 2))}%`,
      tone: 'green'
    },
    {
      label: 'POSE QUEUE',
      value: poseQueue?.toFixed(0) ?? '—',
      unit: 'FRAMES',
      detail: 'Capture → pose channel depth',
      width: `${Math.min(100, (poseQueue ?? 0) * 25)}%`,
      tone: 'orange'
    },
    {
      label: 'OUTPUT QUEUE',
      value: outputQueue?.toFixed(0) ?? '—',
      unit: 'FRAMES',
      detail: 'Pose → output channel depth',
      width: `${Math.min(100, (outputQueue ?? 0) * 25)}%`,
      tone: 'orange'
    },
    {
      label: 'POOL AVAILABLE',
      value: pool?.toFixed(0) ?? '—',
      unit: 'BUFFERS',
      detail: 'Capture pool available',
      width: `${Math.min(100, (pool ?? 0) * 25)}%`,
      tone: 'green'
    },
    {
      label: 'DROPPED FRAMES',
      value: dropped.toFixed(0),
      unit: 'TOTAL',
      detail: 'Capture and pose transport drops',
      width: `${dropped ? 100 : 4}%`,
      tone: dropped ? 'orange' : 'green'
    }
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
        <span class="eyebrow">PIPELINE TELEMETRY</span>
        <h2>Runtime performance</h2>
        <p>Live health and throughput for the current capture session.</p>
      </div>
      <span class="healthy" :class="{ waiting: !snapshot }">{{
        snapshot ? 'LIVE' : 'WAITING FOR IRIS'
      }}</span>
    </header>
    <div class="metrics-cards">
      <article v-for="metric in metrics" :key="metric.label" class="metric-card">
        <span class="metric-label">{{ metric.label }}</span
        ><strong
          >{{ metric.value }} <small>{{ metric.unit }}</small></strong
        ><span class="metric-detail">{{ metric.detail }}</span>
        <div class="large-meter"><i :class="metric.tone" :style="{ width: metric.width }"></i></div>
      </article>
    </div>
    <section class="pipeline-timing" aria-labelledby="pipeline-timing-title">
      <div class="card-title" id="pipeline-timing-title">PIPELINE STAGE TIMING <span>MEAN COMPLETION TIME · MS</span></div>
      <div class="pipeline-flow">
        <article v-for="(stage, index) in stageTimings" :key="stage.name" class="stage-block">
          <div class="stage-heading"><span class="stage-index">0{{ index + 1 }}</span><span class="stage-name">{{ stage.name }}</span></div>
          <strong class="stage-value">{{ stage.value?.toFixed(1) ?? '—' }} <small>MS</small></strong>
          <span class="stage-description">{{ stage.description }}</span>
          <div class="stage-meter"><i :class="stage.tone" :style="{ width: stage.width }"></i></div>
          <span v-if="index < stageTimings.length - 1" class="stage-arrow" aria-hidden="true">→</span>
        </article>
      </div>
    </section>
    <section class="gauge-section">
      <div class="card-title">
        GAUGE HISTORY
        <span>{{ gaugeCharts.length }} LIVE DIAGRAMS · LAST {{ historyLimit / 2 }} SECONDS</span>
      </div>
      <label class="gauge-filter">
        <span>Filter gauges</span>
        <input v-model="gaugeFilter" type="search" placeholder="e.g. queue, camera, pose" />
      </label>
      <div v-if="gaugeCharts.length" class="gauge-grid">
        <article v-for="gauge in gaugeCharts" :key="gauge.name" class="gauge-chart">
          <header>
            <code>{{ gauge.name }}</code
            ><strong>{{ format(gauge.current) }}</strong>
          </header>
          <svg
            viewBox="0 0 100 54"
            preserveAspectRatio="none"
            role="img"
            :aria-label="`${gauge.name} history`"
          >
            <line x1="0" y1="50" x2="100" y2="50" />
            <polyline :points="chartPoints(gauge.values)" />
          </svg>
          <footer>
            <span>{{ format(Math.min(...gauge.values)) }}</span
            ><span>{{ format(Math.max(...gauge.values)) }}</span>
          </footer>
        </article>
      </div>
      <p v-else class="empty-metrics">No gauges match this filter.</p>
    </section>
    <section class="all-metrics">
      <div class="card-title">
        ALL EXPORTED METRICS <span>{{ allMetrics.length }} LIVE SERIES</span>
      </div>
      <div v-if="allMetrics.length" class="metric-table" role="table">
        <div class="metric-row metric-header" role="row">
          <span>Metric</span><span>Type</span><span>Value</span><span>Details</span>
        </div>
        <div v-for="metric in allMetrics" :key="metric.name" class="metric-row" role="row">
          <code>{{ metric.name }}</code
          ><span>{{ metric.kind }}</span
          ><strong>{{ metric.value }}</strong
          ><span>{{ metric.detail }}</span>
        </div>
      </div>
      <p v-else class="empty-metrics">Waiting for the runtime to publish metrics.</p>
    </section>
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
</style>
