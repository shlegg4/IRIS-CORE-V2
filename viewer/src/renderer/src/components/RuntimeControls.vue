<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { CalibrationSnapshot, MetricsSnapshot, RuntimeLogEntry, RuntimeStatus } from '../types/iris'
import CameraManager from './CameraManager.vue'

const props = defineProps<{
  status: RuntimeStatus
  metrics: MetricsSnapshot | null
  logs: RuntimeLogEntry[]
}>()
const emit = defineEmits<{
  (event: 'calibration-refresh', value: CalibrationSnapshot | null): void
}>()

const api = window.api
const busy = ref('')
const message = ref('')
const metricPrefix = ref('')
const logContainer = ref<HTMLElement | null>(null)
const settingsContent = ref<HTMLElement | null>(null)
const activeSection = ref('camera-source')
let sectionObserver: IntersectionObserver | undefined
const selectedCameraId = ref<number | null>(null)
const cameraResolution = ref('1920x1080')
const cameraFrameRate = ref(60)
const cameraFormat = ref('mjpeg')
const configuredCameras = computed(() => props.status.cameras ?? [])
const selectedCamera = computed(() => configuredCameras.value.find((camera) => camera.camera_id === selectedCameraId.value) ?? null)
const displayedFrameRate = computed(() => selectedCamera.value ? cameraFrameRate.value : 60)
const inputLabel = computed(() => props.status.input_mode === 'video' ? 'Video files' : 'Live cameras')

const settingsSections = [
  { id: 'camera-source', label: 'Camera Source' },
  { id: 'pose-estimation', label: 'Pose Estimation' },
  { id: 'calibration', label: 'Calibration' },
  { id: 'recording', label: 'Recording' },
  { id: 'outputs', label: 'Outputs' },
  { id: 'advanced', label: 'Advanced' }
]

const pose = ref({
  backend: 'off',
  model_path: '@assets/pear_ehm_libtorch.pt',
  engine_path: '@assets/rtmo_s.engine',
  calibration_path: ''
})
const recording = ref({ destination: 'recordings/iris-recording.mp4', bitrate: 8000000, frame_rate: 30 })
const calibration = ref({ output_path: 'rig-calibration.json' })
const sync = ref({ tolerance_ms: 20, queue_capacity: 4, incomplete_batch_policy: 'drop' })
const shm = ref({ enabled: false, destination: 'Local\\IRIS_V2_Output', capacity_bytes: 67108864, legacy_v1: true })
const preview = ref({ http_enabled: true, mjpeg_enabled: true, h264_enabled: true, bind_address: '127.0.0.1', port: 8080, max_fps: 30, max_width: 1280, jpeg_quality: 75, bitrate: 4000000, queue_capacity: 16 })

function loadCameraSettings(): void {
  if (!selectedCamera.value) {
    cameraResolution.value = '1920x1080'
    cameraFrameRate.value = 60
    cameraFormat.value = 'mjpeg'
    return
  }
  cameraResolution.value = `${selectedCamera.value.width}x${selectedCamera.value.height}`
  cameraFrameRate.value = selectedCamera.value.frame_rate?.value ?? selectedCamera.value.fps ?? 60
  cameraFormat.value = selectedCamera.value.format ?? 'mjpeg'
}

watch(() => props.status.cameras?.map((camera) => `${camera.camera_id}:${camera.width}x${camera.height}:${camera.frame_rate?.value ?? camera.fps}:${camera.format}`).join('|'), () => {
  if (!configuredCameras.value.some((camera) => camera.camera_id === selectedCameraId.value))
    selectedCameraId.value = configuredCameras.value[0]?.camera_id ?? null
  loadCameraSettings()
}, { immediate: true })
watch(selectedCameraId, loadCameraSettings)

const runtimeState = () => props.status.state || 'Unknown'
const calibrationState = () => props.status.calibration ? 'Ready' : 'Required'

async function call(name: string, action: () => Promise<unknown>): Promise<void> {
  busy.value = name
  message.value = ''
  try {
    await action()
    message.value = 'Applied'
  } catch (error) {
    message.value = error instanceof Error ? error.message : String(error)
  } finally {
    busy.value = ''
  }
}

function patch(path: string, body: unknown): () => Promise<unknown> {
  return () => api.request(path, 'PATCH', body)
}

async function applyCameraSettings(): Promise<void> {
  if (selectedCameraId.value === null) return
  const [width, height] = cameraResolution.value.split('x').map(Number)
  await call('camera-settings', () => api.configureCamera(selectedCameraId.value as number, {
    width,
    height,
    frame_rate: cameraFrameRate.value,
    format: cameraFormat.value
  }))
}

function setCameraFrameRate(event: Event): void {
  cameraFrameRate.value = Number((event.target as HTMLSelectElement).value)
}

async function refreshCalibration(): Promise<void> {
  await call('calibration', async () => {
    const response = await api.getCalibration() as { calibration?: CalibrationSnapshot | null }
    emit('calibration-refresh', response.calibration ?? null)
  })
}

async function clearCalibration(): Promise<void> {
  if (window.confirm('Clear the current rig calibration?'))
    await call('calibration', api.clearCalibration)
}

async function shutdown(): Promise<void> {
  if (window.confirm('Shutdown the IRIS runtime?'))
    await call('runtime', api.shutdownRuntime)
}

function navigateTo(id: string): void {
  activeSection.value = id
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

function updateActiveSection(): void {
  const root = settingsContent.value
  if (!root) return
  if (root.scrollTop + root.clientHeight >= root.scrollHeight - 2) {
    activeSection.value = settingsSections[settingsSections.length - 1].id
    return
  }
  const marker = root.getBoundingClientRect().top + Math.min(120, root.clientHeight * 0.18)
  const current = settingsSections
    .map(({ id }) => document.getElementById(id))
    .filter((section): section is HTMLElement => section instanceof HTMLElement && section.getBoundingClientRect().top <= marker)
    .at(-1)
  activeSection.value = current?.id ?? settingsSections[0].id
}

watch(() => props.logs.length, async () => {
  await nextTick()
  if (logContainer.value)
    logContainer.value.scrollTop = logContainer.value.scrollHeight
})

onMounted(() => {
  if (!settingsContent.value) return
  settingsContent.value.addEventListener('scroll', updateActiveSection, { passive: true })
  if (typeof IntersectionObserver === 'undefined') {
    updateActiveSection()
    return
  }
  sectionObserver = new IntersectionObserver(updateActiveSection, { root: settingsContent.value, threshold: [0, 0.15, 0.4] })
  settingsSections.forEach(({ id }) => {
    const element = document.getElementById(id)
    if (element) sectionObserver?.observe(element)
  })
  updateActiveSection()
})

onBeforeUnmount(() => {
  sectionObserver?.disconnect()
  settingsContent.value?.removeEventListener('scroll', updateActiveSection)
})
</script>

<template>
  <div class="runtime-controls settings-layout">
    <aside class="settings-nav" aria-label="Settings sections">
      <nav>
        <button
          v-for="section in settingsSections"
          :key="section.id"
          :class="{ active: activeSection === section.id }"
          :aria-current="activeSection === section.id ? 'location' : undefined"
          @click="navigateTo(section.id)"
        >{{ section.label }}</button>
      </nav>
      <div class="settings-nav-runtime">
        <span class="settings-state-dot" :class="{ online: props.status.api_connected }"></span>
        <span>{{ props.status.api_connected ? 'Runtime connected' : 'Runtime disconnected' }}</span>
      </div>
    </aside>

    <main ref="settingsContent" class="settings-content">
      <p v-if="message" class="settings-feedback" :class="{ error: message !== 'Applied' }" role="status">{{ message }}</p>

      <section id="camera-source" class="settings-section">
        <header class="settings-section-header">
          <div><h2>Camera Source</h2><p>Choose live cameras or synchronized video files.</p></div>
        </header>
        <div class="camera-source-summary">
          <div><span>Number of Cameras</span><strong>{{ props.status.input_mode === 'video' ? props.status.video_inputs?.length ?? 0 : configuredCameras.length }}</strong></div>
          <div><span>Camera Source</span><strong>{{ inputLabel }}</strong></div>
        </div>
        <div class="settings-form-grid camera-capture-form">
          <label class="settings-field"><span>Camera</span><select v-model.number="selectedCameraId" :disabled="!configuredCameras.length || props.status.input_mode === 'video' || !!busy"><option v-for="camera in configuredCameras" :key="camera.camera_id" :value="camera.camera_id">Camera {{ camera.camera_id + 1 }}</option></select></label>
          <label class="settings-field"><span>Resolution</span><select v-model="cameraResolution" :disabled="!selectedCamera || props.status.input_mode === 'video' || !!busy"><option v-if="![ '1920x1080', '1280x720', '640x480' ].includes(cameraResolution)" :value="cameraResolution">{{ cameraResolution.replace('x', ' × ') }}</option><option value="1920x1080">1920 × 1080</option><option value="1280x720">1280 × 720</option><option value="640x480">640 × 480</option></select></label>
          <label class="settings-field"><span>Frame Rate</span><select :value="displayedFrameRate" :disabled="!selectedCamera || props.status.input_mode === 'video' || !!busy" @change="setCameraFrameRate"><option v-if="![15, 24, 30, 60].includes(displayedFrameRate)" :value="displayedFrameRate">{{ displayedFrameRate }} FPS</option><option :value="15">15 FPS</option><option :value="24">24 FPS</option><option :value="30">30 FPS</option><option :value="60">60 FPS</option></select></label>
          <label class="settings-field"><span>Color Format</span><select v-model="cameraFormat" :disabled="!selectedCamera || props.status.input_mode === 'video' || !!busy"><option value="mjpeg">MJPEG</option><option value="yuy2">YUY2</option><option value="bgra8">BGRA8</option></select></label>
        </div>
        <p v-if="props.status.input_mode === 'video'" class="camera-source-note">Resolution and frame rate come from the selected video files.</p>
        <div class="settings-actions"><button class="settings-primary" :disabled="!selectedCamera || props.status.input_mode === 'video' || !!busy" @click="applyCameraSettings">{{ busy === 'camera-settings' ? 'Applying…' : 'Apply Capture Settings' }}</button></div>
        <details class="settings-subsection camera-management">
          <summary>Manage Cameras and Video Files <span>{{ configuredCameras.length }} configured</span></summary>
          <CameraManager :status="props.status" />
        </details>
      </section>

      <section id="pose-estimation" class="settings-section">
        <header class="settings-section-header">
          <div><h2>Pose Estimation</h2><p>Configure the inference backend and model files.</p></div>
          <span class="settings-current-value">{{ props.status.pose_backend || 'Off' }}</span>
        </header>
        <div class="settings-form-grid">
          <label class="settings-field"><span>Pose backend</span><select v-model="pose.backend" :disabled="!!busy"><option value="off">Off</option><option value="monocular">Monocular</option><option value="2d">2D</option><option value="multiview">Multiview</option></select></label>
          <label class="settings-field"><span>Model path</span><input v-model="pose.model_path" :disabled="!!busy" /></label>
          <label class="settings-field"><span>Engine path</span><input v-model="pose.engine_path" :disabled="!!busy" /></label>
          <label class="settings-field"><span>Calibration file</span><input v-model="pose.calibration_path" :disabled="!!busy" placeholder="Use the active rig calibration" /></label>
        </div>
        <div class="settings-actions"><button class="settings-primary" :disabled="!!busy" @click="call('pose', () => api.configurePose({ ...pose }))">{{ busy === 'pose' ? 'Applying…' : 'Apply Pose Settings' }}</button></div>
      </section>

      <section id="calibration" class="settings-section">
        <header class="settings-section-header">
          <div><h2>Calibration</h2><p>Set up the camera rig for accurate 3D reconstruction.</p></div>
          <span class="calibration-indicator" :class="{ ready: props.status.calibration }"><i></i>{{ calibrationState() }}</span>
        </header>
        <div class="settings-form-grid calibration-form">
          <label class="settings-field"><span>Calibration output file</span><input v-model="calibration.output_path" :disabled="!!busy" /></label>
          <div v-if="props.status.calibration_tool" class="calibration-runtime-state"><span>{{ props.status.calibration_tool.state }}</span><small v-if="props.status.calibration_tool.message">{{ props.status.calibration_tool.message }}</small></div>
        </div>
        <div class="settings-actions">
          <button class="settings-primary" :disabled="!!busy || props.status.state !== 'running'" @click="call('calibration', () => api.startCalibration({ ...calibration }))">{{ busy === 'calibration' ? 'Calibrating…' : 'Recalibrate' }}</button>
          <button :disabled="!!busy" @click="call('calibration', api.cancelCalibration)">Cancel</button>
          <button :disabled="!!busy" @click="refreshCalibration">Refresh</button>
          <button class="settings-danger" :disabled="!!busy" @click="clearCalibration">Clear Calibration</button>
        </div>
      </section>

      <section id="recording" class="settings-section">
        <header class="settings-section-header"><div><h2>Recording</h2><p>Capture and save a session.</p></div><span class="settings-current-value">{{ props.status.recording ? 'Active' : 'Inactive' }}</span></header>
        <div class="settings-form-grid">
          <label class="settings-field"><span>Destination</span><input v-model="recording.destination" :disabled="!!busy" /></label>
          <label class="settings-field"><span>Bitrate (bps)</span><input v-model.number="recording.bitrate" type="number" min="1" :disabled="!!busy" /></label>
          <label class="settings-field"><span>Frame rate (FPS)</span><input v-model.number="recording.frame_rate" type="number" min="1" :disabled="!!busy" /></label>
        </div>
        <div class="settings-actions">
          <button class="settings-primary" :disabled="!!busy || !!props.status.recording" @click="call('recording', () => api.startRecording({ ...recording }))">Start Recording</button>
          <button :disabled="!!busy || !props.status.recording" @click="call('recording', api.stopRecording)">Stop Recording</button>
        </div>
      </section>

      <section id="outputs" class="settings-section">
        <header class="settings-section-header"><div><h2>Outputs</h2><p>Configure preview streams and shared memory.</p></div></header>
        <details class="settings-subsection" open>
          <summary>Preview Stream <span>{{ props.status.preview?.connected_clients || 0 }} clients</span></summary>
          <div class="settings-form-grid">
            <label class="settings-field"><span>Bind address</span><input v-model="preview.bind_address" :disabled="!!busy" /></label>
            <label class="settings-field"><span>Port</span><input v-model.number="preview.port" type="number" min="1" max="65535" :disabled="!!busy" /></label>
            <label class="settings-field"><span>Maximum FPS</span><input v-model.number="preview.max_fps" type="number" min="1" :disabled="!!busy" /></label>
            <label class="settings-field"><span>Maximum width</span><input v-model.number="preview.max_width" type="number" min="1" :disabled="!!busy" /></label>
            <label class="settings-field"><span>JPEG quality</span><input v-model.number="preview.jpeg_quality" type="number" min="1" max="100" :disabled="!!busy" /></label>
            <label class="settings-field"><span>H.264 bitrate</span><input v-model.number="preview.bitrate" type="number" min="1" :disabled="!!busy" /></label>
            <label class="settings-field"><span>Queue capacity</span><input v-model.number="preview.queue_capacity" type="number" min="1" :disabled="!!busy" /></label>
          </div>
          <div class="settings-toggle-row"><label><input v-model="preview.http_enabled" type="checkbox" :disabled="!!busy" /> HTTP</label><label><input v-model="preview.mjpeg_enabled" type="checkbox" :disabled="!!busy" /> MJPEG</label><label><input v-model="preview.h264_enabled" type="checkbox" :disabled="!!busy" /> H.264</label></div>
          <div class="settings-actions"><button class="settings-primary" :disabled="!!busy" @click="call('preview', patch('/outputs/preview', { ...preview }))">Apply Preview Settings</button></div>
        </details>
        <details class="settings-subsection">
          <summary>Shared Memory <span>{{ props.status.shared_memory_enabled ? 'Enabled' : 'Disabled' }}</span></summary>
          <div class="settings-form-grid">
            <label class="settings-field"><span>Destination</span><input v-model="shm.destination" :disabled="!!busy" /></label>
            <label class="settings-field"><span>Capacity (bytes)</span><input v-model.number="shm.capacity_bytes" type="number" min="1" :disabled="!!busy" /></label>
          </div>
          <div class="settings-toggle-row"><label><input v-model="shm.enabled" type="checkbox" :disabled="!!busy" /> Enabled</label><label><input v-model="shm.legacy_v1" type="checkbox" :disabled="!!busy" /> Publish legacy v1</label></div>
          <div class="settings-actions"><button class="settings-primary" :disabled="!!busy" @click="call('shared-memory', patch('/outputs/shared-memory', { ...shm }))">Apply Shared Memory</button></div>
        </details>
      </section>

      <section id="advanced" class="settings-section">
        <header class="settings-section-header"><div><h2>Advanced</h2><p>Runtime, synchronization and diagnostic controls.</p></div></header>
        <details class="settings-subsection" open>
          <summary>Synchronizer <span>{{ props.status.sync_tolerance_ms ?? sync.tolerance_ms }} ms tolerance</span></summary>
          <div class="settings-form-grid">
            <label class="settings-field"><span>Tolerance (ms)</span><input v-model.number="sync.tolerance_ms" type="number" min="0" :disabled="!!busy" /></label>
            <label class="settings-field"><span>Queue capacity</span><input v-model.number="sync.queue_capacity" type="number" min="1" :disabled="!!busy" /></label>
            <label class="settings-field"><span>Incomplete batch policy</span><select v-model="sync.incomplete_batch_policy" :disabled="!!busy"><option value="drop">Drop incomplete batches</option><option value="partial">Emit partial batches</option></select></label>
          </div>
          <div class="settings-actions"><button class="settings-primary" :disabled="!!busy" @click="call('synchronizer', patch('/synchronizer', { ...sync }))">Apply Synchronizer</button></div>
        </details>
        <details class="settings-subsection">
          <summary>Metrics Export <span>{{ Object.keys(props.metrics?.counters || {}).length }} counters · {{ Object.keys(props.metrics?.gauges || {}).length }} gauges</span></summary>
          <div class="settings-form-grid settings-metrics-form">
            <label class="settings-field"><span>Metric prefix</span><input v-model="metricPrefix" aria-label="Metric prefix" placeholder="All metrics" :disabled="!!busy" /></label>
          </div>
          <div class="settings-actions"><button :disabled="!!busy" @click="call('metrics', () => api.getMetrics(metricPrefix))">Refresh Metrics</button></div>
        </details>
        <details class="settings-subsection">
          <summary>Runtime <span>{{ runtimeState() }}</span></summary>
          <div class="runtime-action-row">
            <span>Status <strong>{{ runtimeState() }}</strong><small v-if="props.status.last_error">{{ props.status.last_error }}</small></span>
            <button class="settings-primary" :disabled="!!busy" @click="call('runtime', api.startPipeline)">Start Pipeline</button>
            <button :disabled="!!busy" @click="call('runtime', api.stopPipeline)">Stop Pipeline</button>
            <button :disabled="!!busy" @click="call('runtime', api.getStatus)">Refresh Status</button>
            <button class="settings-danger" :disabled="!!busy" @click="shutdown">Shutdown</button>
          </div>
        </details>
        <details class="settings-subsection">
          <summary>Terminal <span>{{ props.logs.length }} events</span></summary>
          <div ref="logContainer" class="control-log" role="log">
            <div v-for="log in props.logs" :key="log.id" :class="{ 'control-log-error': log.level === 'error' }">{{ new Date(log.timestamp).toLocaleTimeString('en-GB') }} · {{ log.source }} · {{ log.message }}</div>
            <small v-if="!props.logs.length">No runtime events yet.</small>
          </div>
        </details>
      </section>
    </main>
  </div>
</template>
