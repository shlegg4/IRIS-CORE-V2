<script setup lang="ts">
import { computed } from 'vue'
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
const metrics = computed(() => {
  const interval = average('iris_capture_interframe_interval_ms')
  const captureRate = interval && interval > 0 ? 1000 / interval : null
  const inference = average('iris_pose_inference_ms') ?? average('iris_multiview_pose_inference_ms')
  const confidence = metricValue(['iris_pose_confidence', 'iris_multiview_pose_confidence'])
  const pool = metricValue(['iris_capture_pool_available'])
  return [
    {
      label: 'CAPTURE RATE',
      value: captureRate?.toFixed(1) ?? '—',
      unit: 'FPS',
      detail: '4 synchronized sources',
      width: '96%',
      tone: 'green'
    },
    {
      label: 'INFERENCE LATENCY',
      value: inference?.toFixed(1) ?? '—',
      unit: 'MS',
      detail: 'RTMO-S pose pipeline',
      width: '72%',
      tone: 'cyan'
    },
    {
      label: 'TRIANGULATION',
      value: confidence == null ? '—' : (confidence * (confidence <= 1 ? 100 : 1)).toFixed(1),
      unit: '%',
      detail: 'Mean confidence across joints',
      width: '98%',
      tone: 'green'
    },
    {
      label: 'GPU MEMORY',
      value: pool?.toFixed(0) ?? '—',
      unit: 'BUFFERS',
      detail: 'Capture pool available',
      width: `${Math.min(100, (pool ?? 0) * 25)}%`,
      tone: 'orange'
    }
  ]
})
const events = [
  'Frame sync restored',
  'GPU memory pool allocated',
  'Camera 04 exposure adjusted',
  'Pose model warmed'
]
</script>

<template>
  <div class="metrics-page">
    <header class="metrics-hero">
      <div>
        <span class="eyebrow">PIPELINE TELEMETRY</span>
        <h2>Runtime performance</h2>
        <p>Live health and throughput for the current capture session.</p>
      </div>
      <span class="healthy">HEALTHY</span>
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
    <div class="metric-lower">
      <section class="chart-card">
        <div class="card-title">FRAME TIME <span>LAST 30 SECONDS</span></div>
        <div class="bars">
          <i v-for="n in 30" :key="n" :style="{ height: `${24 + ((n * 17) % 48)}%` }"></i>
        </div>
        <div class="chart-axis"><span>16.8 ms</span><span>14.2 ms</span><span>12.0 ms</span></div>
      </section>
      <section class="events-card">
        <div class="card-title">RECENT EVENTS <span>LIVE</span></div>
        <div v-for="event in events" :key="event" class="event">
          <i></i><span>{{ event }}</span
          ><time>10:42:08</time>
        </div>
      </section>
    </div>
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
.chart-card,
.events-card {
  min-width: 0;
  padding: 18px;
  border: 1px solid #1e292d;
  background: #10181b;
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
.metric-lower {
  display: grid;
  grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr);
  gap: 12px;
  padding-bottom: 1px;
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
.bars {
  height: 180px;
  display: flex;
  align-items: end;
  gap: 5px;
  overflow: hidden;
  border-bottom: 1px solid #26373b;
}
.bars i {
  min-width: 2px;
  min-height: 12px;
  flex: 1 1 0;
  background: #3f999e;
}
.chart-axis {
  display: flex;
  justify-content: space-between;
  margin-top: 8px;
  color: #718087;
  font-size: 9px;
}
.event {
  min-width: 0;
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 13px 0;
  border-bottom: 1px solid #1e292d;
  color: #a9b8b9;
}
.event > i {
  width: 6px;
  height: 6px;
  flex: 0 0 6px;
  margin-top: 4px;
  border-radius: 50%;
  background: #79dbb0;
}
.event > span {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.event time {
  flex: 0 0 auto;
  margin-left: auto;
  color: #526368;
  font-size: 9px;
}
@media (max-width: 1000px) {
  .metrics-cards {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
@media (max-width: 760px) {
  .metrics-page {
    padding: 20px 18px 28px;
  }
  .metric-lower {
    grid-template-columns: 1fr;
  }
}
@media (max-width: 480px) {
  .metrics-cards {
    grid-template-columns: 1fr;
  }
  .metrics-hero p {
    display: none;
  }
}
</style>
