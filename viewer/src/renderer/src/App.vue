<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import MetricsPage from './components/MetricsPage.vue'
import CameraGridPage from './components/CameraGridPage.vue'
import RuntimeControls from './components/RuntimeControls.vue'
import ThreeGraph from './components/ThreeGraph.vue'
import ViewerSidebar from './components/ViewerSidebar.vue'
import type { CameraRotation, CalibrationSnapshot, MetricsSnapshot, PoseFrame, RuntimeLogEntry, RuntimeStatus } from './types/iris'

type Page = 'view' | 'metrics' | 'settings'
const activePage = ref<Page>('view')
const showGrid = ref(true)
const showCameras = ref(true)
const logs = ref<RuntimeLogEntry[]>([])
const metrics = ref<MetricsSnapshot | null>(null)
const poseFrame = ref<PoseFrame | null>(null)
const runtimeStatus = ref<RuntimeStatus>({})
const calibration = ref<CalibrationSnapshot | null>(null)
const rotationBusy = ref<number | null>(null)
const rotationError = ref('')
const api = window.api
const poseSequence = computed(() => poseFrame.value?.sequence ?? poseFrame.value?.sourceSequence ?? 0)
const posePointCount = computed(() => poseFrame.value?.people?.reduce((total, person) =>
  total + (person.valid ?? person.jointValid ?? []).filter(Boolean).length, 0) ??
  (poseFrame.value?.joints ?? poseFrame.value?.joints3d ?? poseFrame.value?.joints_3d)?.length ?? 0)
const hasPoseData = computed(() => posePointCount.value > 0 || !!poseFrame.value)
const cameras = computed(() => runtimeStatus.value.input_mode === 'video'
  ? runtimeStatus.value.video_inputs ?? []
  : runtimeStatus.value.cameras ?? [])
const connectedCount = computed(() => runtimeStatus.value.input_mode === 'video'
  ? cameras.value.length
  : (runtimeStatus.value.cameras ?? []).filter((camera) => camera.connected || camera.state === 'connected').length)
const backendLabel = computed(() => {
  const backend = runtimeStatus.value.pose_backend || 'off'
  return backend === '2d' ? '2D' : backend === 'off' ? 'Off' : backend[0].toUpperCase() + backend.slice(1)
})
const calibrationLabel = computed(() => {
  const state = runtimeStatus.value.calibration_tool?.state
  if (state && !['idle', 'ready', 'complete', 'completed'].includes(state.toLowerCase()))
    return state.replaceAll('_', ' ').replace(/^./, (letter) => letter.toUpperCase())
  return runtimeStatus.value.calibration ? 'Calibration ready' : 'Calibration required'
})
const pipelineLabel = computed(() => {
  const state = runtimeStatus.value.state || 'stopped'
  return state === 'running' ? 'Pipeline running' : state === 'failed' || state === 'error' ? 'Pipeline failed' : state === 'starting' ? 'Pipeline starting' : state === 'stopping' ? 'Pipeline stopping' : 'Pipeline stopped'
})
const sourceLabel = computed(() => runtimeStatus.value.input_mode === 'video' ? 'Video input' : 'Live input')
const pipelineFps = computed(() => {
  const interval = metrics.value?.histograms?.iris_capture_interframe_interval_ms
  return interval?.count && interval.sum > 0 ? (1000 * interval.count / interval.sum) : null
})
const poseLatency = computed(() => {
  const gauges = metrics.value?.gauges ?? {}
  const direct = gauges.iris_pose_last_capture_to_result_ms
  if (typeof direct === 'number') return direct
  const histogram = metrics.value?.histograms?.iris_pose_capture_to_result_ms
  return histogram?.count ? histogram.sum / histogram.count : null
})
const statusFps = computed(() => pipelineFps.value === null ? '— FPS' : `${pipelineFps.value.toFixed(0)} FPS`)
const statusLatency = computed(() => poseLatency.value === null ? '— ms' : `${poseLatency.value.toFixed(1)} ms`)
const subscriptions: Array<() => void> = []

function addLog(entry: RuntimeLogEntry): void {
  if (logs.value.some((existing) => existing.id === entry.id)) return
  logs.value = [...logs.value, entry].sort((a, b) => a.id - b.id).slice(-500)
}

function applyCalibrationSnapshot(value: CalibrationSnapshot | null): void {
  calibration.value = value
  runtimeStatus.value.calibration = value
}

function rotationFor(cameraId: number): CameraRotation {
  if (runtimeStatus.value.input_mode === 'video')
    return runtimeStatus.value.video_inputs?.find((feed) => feed.camera_id === cameraId)?.rotation ?? 'none'
  return runtimeStatus.value.cameras?.find((camera) => camera.camera_id === cameraId)?.rotation ?? 'none'
}

function nextRotation(rotation: CameraRotation): CameraRotation {
  const order: CameraRotation[] = ['none', 'cw90', '180', 'ccw90']
  return order[(order.indexOf(rotation) + 1) % order.length]
}

async function rotateCamera(cameraId: number): Promise<void> {
  if (rotationBusy.value !== null) return
  rotationBusy.value = cameraId
  rotationError.value = ''
  const rotation = nextRotation(rotationFor(cameraId))
  try {
    if (runtimeStatus.value.input_mode === 'video') {
      await api.request('/video-source', 'POST', {
        cameras: (runtimeStatus.value.video_inputs ?? []).map((feed) => ({
          ...feed,
          rotation: feed.camera_id === cameraId ? rotation : feed.rotation ?? 'none'
        })),
        cuda_device: runtimeStatus.value.video_cuda_device ?? 0,
        frame_pool_capacity: runtimeStatus.value.video_frame_pool_capacity ?? 8,
        realtime: runtimeStatus.value.video_realtime ?? true,
        loop: runtimeStatus.value.video_loop ?? false
      })
    } else {
      await api.configureCamera(cameraId, { rotation })
    }
  } catch (error) {
    rotationError.value = error instanceof Error ? error.message : String(error)
  } finally {
    rotationBusy.value = null
  }
}

onMounted(() => {
  if (!api?.onLog) return
  subscriptions.push(
    api.onLog(addLog),
    api.onMetrics((snapshot) => { metrics.value = snapshot as MetricsSnapshot }),
    api.onPoseFrame((frame) => { poseFrame.value = frame as PoseFrame }),
    api.onStatus((status) => {
      const value = status as RuntimeStatus
      runtimeStatus.value = {
        ...runtimeStatus.value,
        ...value,
        cameras: value.cameras ?? runtimeStatus.value.cameras,
        video_inputs: value.video_inputs ?? runtimeStatus.value.video_inputs,
        preview: value.preview ? { ...runtimeStatus.value.preview, ...value.preview } : runtimeStatus.value.preview
      }
      if (Object.prototype.hasOwnProperty.call(value, 'calibration'))
        calibration.value = value.calibration ?? null
      if (value.metrics) metrics.value = value.metrics as MetricsSnapshot
    })
  )
  void api.getRecentLogs().then((entries) => entries.forEach(addLog))
})
onBeforeUnmount(() => subscriptions.forEach((unsubscribe) => unsubscribe()))
</script>

<template>
  <main class="app-shell">
    <header class="app-topbar">
      <div class="brand-wordmark">IRIS</div>
      <nav class="page-tabs" aria-label="Main pages">
        <button :class="{ active: activePage === 'view' }" :aria-current="activePage === 'view' ? 'page' : undefined" @click="activePage = 'view'">3D View</button>
        <button :class="{ active: activePage === 'metrics' }" :aria-current="activePage === 'metrics' ? 'page' : undefined" @click="activePage = 'metrics'">Metrics</button>
        <button :class="{ active: activePage === 'settings' }" :aria-current="activePage === 'settings' ? 'page' : undefined" @click="activePage = 'settings'">Settings</button>
      </nav>
    </header>

    <section v-if="activePage === 'view'" class="output-layout">
      <ViewerSidebar :status="runtimeStatus" :metrics="metrics" @open-settings="activePage = 'settings'" />
      <section class="live-stage" aria-label="Live pose view">
        <div class="viewport">
          <div class="viewport-title"><i class="live-dot"></i><span>Live 3D Output</span></div>
          <div v-if="!hasPoseData" class="viewport-empty">
            <h1>{{ runtimeStatus.state === 'running' ? 'Waiting for pose data' : 'Pipeline stopped' }}</h1>
            <p v-if="runtimeStatus.last_error">{{ runtimeStatus.last_error }}</p>
            <p v-else>{{ cameras.length }} camera{{ cameras.length === 1 ? '' : 's' }} · {{ backendLabel }} pose</p>
          </div>
          <ThreeGraph :show-grid="showGrid" :show-cameras="showCameras" :pose="poseFrame" :calibration="calibration" />
          <div v-if="hasPoseData" class="viewport-frame">Frame {{ String(poseSequence).padStart(6, '0') }}<span>{{ posePointCount }} points</span></div>
        </div>
        <CameraGridPage
          compact
          :status="runtimeStatus"
          :rotation-busy="rotationBusy"
          @rotate="rotateCamera"
        />
        <p v-if="rotationError" class="rotation-error" role="alert">{{ rotationError }}</p>
      </section>
    </section>

    <section v-show="activePage === 'metrics'" class="single-page">
      <MetricsPage
        :snapshot="metrics"
        :input-mode="runtimeStatus.input_mode"
        :video-decode-status="runtimeStatus.video_decode_status"
        :active-camera-count="connectedCount"
        :camera-count="cameras.length"
      />
    </section>

    <section v-if="activePage === 'settings'" class="settings-page">
      <RuntimeControls :status="runtimeStatus" :metrics="metrics" :logs="logs" @calibration-refresh="applyCalibrationSnapshot" />
    </section>

    <footer class="statusbar" aria-label="Runtime status">
      <span class="status-item"><i class="status-dot" :class="runtimeStatus.api_connected ? 'online' : 'offline'"></i>{{ runtimeStatus.api_connected ? 'Runtime connected' : 'Runtime disconnected' }}</span>
      <span class="status-divider"></span>
      <span class="status-item" :class="{ 'status-warning': runtimeStatus.state === 'failed' || runtimeStatus.state === 'error' }">{{ pipelineLabel }}</span>
      <span class="status-divider"></span>
      <span>{{ connectedCount }}/{{ cameras.length }} cameras</span>
      <span>·</span>
      <span>{{ sourceLabel }}</span>
      <span>·</span>
      <span>{{ backendLabel }}</span>
      <span>·</span>
      <span :class="{ 'status-warning': !runtimeStatus.calibration }">{{ calibrationLabel }}</span>
      <span v-if="runtimeStatus.recording" class="recording-state">Recording</span>
      <span class="status-spacer"></span>
      <span class="status-metric">{{ statusFps }}</span>
      <span>·</span>
      <span class="status-metric">{{ statusLatency }}</span>
    </footer>
  </main>
</template>
