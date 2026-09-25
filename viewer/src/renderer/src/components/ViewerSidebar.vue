<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { CameraStatus, MetricsSnapshot, RuntimeStatus } from '../types/iris'

const props = defineProps<{
  status: RuntimeStatus
  metrics: MetricsSnapshot | null
}>()
const emit = defineEmits<{
  openSettings: []
}>()

const api = window.api
const selectedCameraId = ref<number | null>(null)
const resolution = ref('1920x1080')
const frameRate = ref(30)
const busy = ref(false)
const message = ref('')

const cameras = computed<CameraStatus[]>(() => {
  if (props.status.input_mode === 'video') {
    return (props.status.video_inputs ?? []).map((feed) => {
      const configured = props.status.cameras?.find((camera) => camera.camera_id === feed.camera_id)
      return {
        camera_id: feed.camera_id,
        width: configured?.width ?? 1920,
        height: configured?.height ?? 1080,
        fps: configured?.fps ?? 0,
        rotation: feed.rotation,
        reconnect: false,
        ...configured
      }
    })
  }
  return props.status.cameras ?? []
})

const selectedCamera = computed(
  () => cameras.value.find((camera) => camera.camera_id === selectedCameraId.value) ?? null
)
const selectedResolution = computed(() =>
  selectedCamera.value ? `${selectedCamera.value.width}x${selectedCamera.value.height}` : resolution.value
)
const selectedFrameRate = computed(() => {
  const camera = selectedCamera.value
  if (!camera) return frameRate.value
  if (typeof camera.frame_rate === 'number') return camera.frame_rate
  return camera.frame_rate?.value ?? camera.fps ?? frameRate.value
})
const pipelineRunning = computed(() => props.status.state === 'running')
const calibrationStatus = computed(() => {
  const toolState = props.status.calibration_tool?.state
  if (toolState && toolState !== 'idle' && toolState !== 'ready') return toolState.replaceAll('_', ' ')
  return props.status.calibration ? 'Ready' : 'Required'
})

watch(cameras, (list) => {
  if (!list.some((camera) => camera.camera_id === selectedCameraId.value))
    selectedCameraId.value = list[0]?.camera_id ?? null
  const selected = list.find((camera) => camera.camera_id === selectedCameraId.value)
  if (selected) {
    resolution.value = `${selected.width}x${selected.height}`
    frameRate.value = typeof selected.frame_rate === 'number'
      ? selected.frame_rate
      : selected.frame_rate?.value ?? selected.fps ?? 30
  }
}, { immediate: true })

function selectCamera(camera: CameraStatus): void {
  selectedCameraId.value = camera.camera_id
  resolution.value = `${camera.width}x${camera.height}`
  frameRate.value = typeof camera.frame_rate === 'number'
    ? camera.frame_rate
    : camera.frame_rate?.value ?? camera.fps ?? 30
}

async function applyCaptureSettings(): Promise<void> {
  if (!selectedCamera.value || props.status.input_mode === 'video') return
  const [width, height] = resolution.value.split('x').map(Number)
  busy.value = true
  message.value = ''
  try {
    await api.configureCamera(selectedCamera.value.camera_id, {
      width,
      height,
      frame_rate: frameRate.value
    })
    message.value = 'Capture settings applied'
  } catch (error) {
    message.value = error instanceof Error ? error.message : String(error)
  } finally {
    busy.value = false
  }
}

async function togglePipeline(): Promise<void> {
  busy.value = true
  message.value = ''
  try {
    if (pipelineRunning.value) await api.stopPipeline()
    else await api.startPipeline()
  } catch (error) {
    message.value = error instanceof Error ? error.message : String(error)
  } finally {
    busy.value = false
  }
}

async function recalibrate(): Promise<void> {
  busy.value = true
  message.value = ''
  try {
    await api.startCalibration()
  } catch (error) {
    message.value = error instanceof Error ? error.message : String(error)
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <aside class="viewer-sidebar" aria-label="Capture controls">
    <section class="sidebar-section camera-setup">
      <h2>Camera Setup</h2>
      <div v-if="cameras.length" class="camera-source-list" role="listbox" aria-label="Select camera to configure">
        <button
          v-for="camera in cameras"
          :key="camera.camera_id"
          class="camera-source-row"
          :class="{ selected: camera.camera_id === selectedCameraId }"
          role="option"
          :aria-selected="camera.camera_id === selectedCameraId"
          @click="selectCamera(camera)"
        >
          <span class="camera-connection-dot" :class="{ offline: !camera.connected && props.status.input_mode !== 'video' }"></span>
          <span>Cam {{ camera.camera_id + 1 }}</span>
          <span class="camera-row-state">{{ camera.connected || props.status.input_mode === 'video' ? 'Connected' : 'Offline' }}</span>
        </button>
      </div>
      <p v-else class="sidebar-empty">No cameras configured</p>
      <button class="add-camera-button" @click="emit('openSettings')">Add Camera</button>

      <div class="capture-format-fields">
        <label>
          <span>Resolution</span>
          <select v-model="resolution" :disabled="busy || !selectedCamera || props.status.input_mode === 'video'" @change="applyCaptureSettings">
            <option v-if="!['1920x1080','1280x720','640x480'].includes(selectedResolution)" :value="selectedResolution">{{ selectedResolution.replace('x', ' × ') }}</option>
            <option value="1920x1080">1920 × 1080</option>
            <option value="1280x720">1280 × 720</option>
            <option value="640x480">640 × 480</option>
          </select>
        </label>
        <label>
          <span>FPS</span>
          <select v-model.number="frameRate" :disabled="busy || !selectedCamera || props.status.input_mode === 'video'" @change="applyCaptureSettings">
            <option v-if="![15,24,30,60].includes(selectedFrameRate)" :value="selectedFrameRate">{{ selectedFrameRate }} FPS</option>
            <option :value="15">15 FPS</option>
            <option :value="24">24 FPS</option>
            <option :value="30">30 FPS</option>
            <option :value="60">60 FPS</option>
          </select>
        </label>
      </div>
    </section>

    <section class="sidebar-section pose-setup">
      <h2>Pose Estimation</h2>
      <button class="recalibrate-button" :disabled="busy || pipelineRunning !== true" @click="recalibrate">Recalibrate</button>
      <span class="calibration-state" :class="{ required: !props.status.calibration }">
        <i class="state-dot"></i>{{ calibrationStatus }}
      </span>
    </section>

    <section class="sidebar-section capture-controls">
      <h2>Capture</h2>
      <button class="pipeline-button" :class="{ stopping: pipelineRunning }" :disabled="busy || !props.status.api_connected" @click="togglePipeline">
        {{ busy ? 'Please wait…' : pipelineRunning ? 'Stop pipeline' : 'Start pipeline' }}
      </button>
    </section>

    <p v-if="message" class="sidebar-message" :class="{ error: !message.endsWith('applied') }" role="status">{{ message }}</p>
  </aside>
</template>
