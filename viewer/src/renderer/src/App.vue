<script setup lang="ts">
import { computed, onMounted, onBeforeUnmount, ref } from 'vue'
import ThreeGraph from './components/ThreeGraph.vue'
import MetricsPage from './components/MetricsPage.vue'
import CameraGridPage from './components/CameraGridPage.vue'
import RuntimeControls from './components/RuntimeControls.vue'
import type { CalibrationSnapshot, MetricsSnapshot, PoseFrame, RuntimeStatus } from './types/iris'
const showGrid = ref(true),
  activeTab = ref<'output' | 'metrics' | 'cameras'>('output')
const logs = ref<string[][]>([])
const emptyStateError = ref('')
const metrics = ref<MetricsSnapshot | null>(null)
const poseFrame = ref<PoseFrame | null>(null)
const calibration = ref<CalibrationSnapshot | null>(null)
const runtimeStatus = ref<RuntimeStatus>({})
const showCameras = ref(true)
const api = window.api
const consoleWidth = ref(340)
const poseSequence = computed(
  () => poseFrame.value?.sequence ?? poseFrame.value?.sourceSequence ?? 0
)
const posePointCount = computed(
  () => poseFrame.value?.people?.reduce((total, person) =>
    total + (person.valid ?? person.jointValid ?? []).filter((valid) => valid).length, 0) ??
    (poseFrame.value?.joints ?? poseFrame.value?.joints3d ?? poseFrame.value?.joints_3d)?.length ?? 0
)
const runtimeState = computed(() => runtimeStatus.value.state || 'unknown')
const runtimeLabel = computed(() => runtimeState.value === 'running' ? 'Running' : runtimeState.value === 'stopped' ? 'Ready' : runtimeState.value[0].toUpperCase() + runtimeState.value.slice(1))
const hasPoseData = computed(() => posePointCount.value > 0 || !!poseFrame.value)
function applyCalibrationSnapshot(value: CalibrationSnapshot | null): void {
  calibration.value = value
}
async function startFromEmpty(): Promise<void> {
  emptyStateError.value = ''
  try {
    await api.startPipeline()
  } catch (error) {
    emptyStateError.value = error instanceof Error ? error.message : String(error)
  }
}
function startResize(event: PointerEvent): void {
  const startX = event.clientX
  const startWidth = consoleWidth.value
  const onMove = (move: PointerEvent): void => {
    const maximum = Math.max(220, window.innerWidth - 360)
    consoleWidth.value = Math.min(maximum, Math.max(300, startWidth + move.clientX - startX))
  }
  const onEnd = (): void => {
    window.removeEventListener('pointermove', onMove)
    window.removeEventListener('pointerup', onEnd)
  }
  window.addEventListener('pointermove', onMove)
  window.addEventListener('pointerup', onEnd, { once: true })
}
const subscriptions: Array<() => void> = []
onMounted(() => {
  subscriptions.push(
    window.api.onLog((log) =>
      logs.value.push([new Date().toLocaleTimeString('en-GB'), 'info', log])
    ),
    window.api.onMetrics((snapshot) => {
      metrics.value = snapshot as MetricsSnapshot
    }),
    window.api.onPoseFrame((frame) => {
      poseFrame.value = frame as PoseFrame
    }),
    window.api.onStatus((status) => {
      const value = status as RuntimeStatus
      runtimeStatus.value = {
        ...runtimeStatus.value,
        ...value,
        cameras: value.cameras ?? runtimeStatus.value.cameras,
        preview: value.preview ? { ...runtimeStatus.value.preview, ...value.preview } : runtimeStatus.value.preview
      }
      if (Object.prototype.hasOwnProperty.call(value, 'calibration'))
        calibration.value = value.calibration ?? null
    })
  )
})
onBeforeUnmount(() => subscriptions.forEach((unsubscribe) => unsubscribe()))
</script>
<template>
  <main class="app-shell">
    <section class="workspace" :style="{ '--console-width': `${consoleWidth}px` }">
      <aside class="console panel"><div class="panel-heading">IRIS CONTROL <span>{{ runtimeLabel }}</span></div><RuntimeControls :status="runtimeStatus" :metrics="metrics" :logs="logs" @calibration-refresh="applyCalibrationSnapshot" /><div class="console-footer">REST · 127.0.0.1:8090 <span>LOCAL</span></div></aside>
      <div
        class="console-resizer"
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize runtime controls"
        tabindex="0"
        @pointerdown="startResize"
      ></div>
      <section class="main-stage">
        <div class="stage-toolbar">
          <div class="tabs">
            <button :class="{ active: activeTab === 'output' }" :aria-pressed="activeTab === 'output'" @click="activeTab = 'output'">
              3D OUTPUT</button
            ><button :class="{ active: activeTab === 'metrics' }" :aria-pressed="activeTab === 'metrics'" @click="activeTab = 'metrics'">
              METRICS</button
            ><button :class="{ active: activeTab === 'cameras' }" :aria-pressed="activeTab === 'cameras'" @click="activeTab = 'cameras'">
              CAMERA GRID
            </button>
          </div>
          <div v-if="activeTab === 'output'" class="stage-tools">
            <button @click="showGrid = !showGrid">{{ showGrid ? '▧ GRID' : '▧ GRID OFF' }}</button
            ><button @click="showCameras = !showCameras">{{ showCameras ? '◎ CAMERAS' : '◎ CAMERAS OFF' }}</button><button>⛶</button>
          </div>
        </div>
        <div v-if="activeTab === 'output'" class="viewport">
          <div v-if="hasPoseData" class="viewport-label"><span class="pill green-pill"><span class="dot green"></span>LIVE</span><span>FRAME {{ String(poseSequence).padStart(6, '0') }}</span><span>{{ runtimeStatus.cameras?.[0]?.width || '—' }} × {{ runtimeStatus.cameras?.[0]?.height || '—' }}</span></div>
          <div v-else class="viewport-empty"><span class="empty-kicker">NO POSE DATA YET</span><h2>{{ runtimeState === 'running' ? 'Waiting for frames' : 'Pipeline is not running' }}</h2><p>{{ runtimeStatus.cameras?.length || 0 }} camera{{ (runtimeStatus.cameras?.length || 0) === 1 ? '' : 's' }} configured · Pose backend: {{ runtimeStatus.pose_backend || 'off' }}</p><small v-if="emptyStateError" class="empty-error" role="alert">{{ emptyStateError }}</small><button v-if="runtimeState !== 'running'" class="empty-primary" @click="startFromEmpty">START PIPELINE</button><button v-else class="empty-secondary" @click="activeTab = 'cameras'">CHECK CAMERAS</button></div>
          <ThreeGraph :show-grid="showGrid" :show-cameras="showCameras" :pose="poseFrame" :calibration="calibration" />
          <div v-if="hasPoseData" class="viewport-hud">
            <span>POSE_ESTIMATION</span><strong>{{ runtimeStatus.pose_backend || '—' }}</strong><span class="hud-divider"></span
            ><span
              >POINTS <strong>{{ posePointCount }}</strong></span
            >
          </div>
        </div>
        <MetricsPage v-else-if="activeTab === 'metrics'" :snapshot="metrics" :input-mode="runtimeStatus.input_mode" :video-decode-status="runtimeStatus.video_decode_status" />
        <CameraGridPage v-if="activeTab === 'cameras'" :status="runtimeStatus" />
      </section>
    </section>
    <footer class="statusbar">
      <span><i :class="['dot', runtimeState === 'running' ? 'green' : runtimeState === 'failed' ? 'red' : 'amber']"></i> {{ runtimeLabel.toUpperCase() }}</span><span v-if="runtimeStatus.processed_packets !== undefined">PACKETS {{ runtimeStatus.processed_packets }}</span><span v-if="runtimeStatus.recording">RECORDING</span><span class="footer-right">IRIS RUNTIME · LOCAL</span>
    </footer>
  </main>
</template>
