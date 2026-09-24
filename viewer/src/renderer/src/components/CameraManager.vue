<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import type {
  CameraStatus,
  CameraRotation,
  CreateCameraRequest,
  DiscoveredCamera,
  RuntimeStatus
} from '../types/iris'

const props = defineProps<{ status: RuntimeStatus }>()
const api = window.api
type VideoFeed = { camera_id: number; path: string; rotation: CameraRotation }
type CameraDraft = {
  camera_id: number
  device_index: number | null
  device_symbolic_link: string
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

const blank = (): CameraDraft => ({
  camera_id: 0,
  device_index: 0,
  device_symbolic_link: '',
  width: 1920,
  height: 1080,
  frame_rate: 30,
  format: 'mjpeg',
  cuda_device: 0,
  sample_queue_capacity: 1,
  frame_pool_capacity: 4,
  overflow: 'drop-oldest',
  rotation: 'none',
  allow_format_fallback: false,
  reconnect: true
})

const editingId = ref<number | null>(null)
const draft = ref<CameraDraft>(blank())
const saved = ref<CameraDraft>(blank())
const mode = ref<'closed' | 'add' | 'edit'>('closed')
const busy = ref(false)
const error = ref('')
const notice = ref('')
const scanBusy = ref(false)
const scanError = ref('')
const scanComplete = ref(false)
const discovered = ref<DiscoveredCamera[]>([])
const addingKey = ref<string | null>(null)
const rotationBusy = ref<string | null>(null)
const rotationErrors = ref<Record<string, string>>({})
const rotationSelections = ref<Record<string, CameraRotation>>({})
const enabledLocally = ref<Record<string, number>>({})
const sourcePanel = ref<'live' | 'video'>(props.status.input_mode === 'video' ? 'video' : 'live')
const videoFeeds = ref<VideoFeed[]>([])
const videoCudaDevice = ref(0)
const videoFramePool = ref(8)
const videoRealtime = ref(false)
const videoLoop = ref(false)
const videoBusy = ref(false)
const videoError = ref('')
const videoNotice = ref('')
const videoDirty = ref(false)

const cameras = computed(() => props.status.cameras || [])
const dirty = computed(() => JSON.stringify(draft.value) !== JSON.stringify(saved.value))
const connectedCount = computed(
  () => cameras.value.filter((camera) => camera.connected || camera.state === 'connected').length
)
const errorCount = computed(
  () => cameras.value.filter((camera) => Boolean(camera.last_error) || (camera.source_errors ?? 0) > 0).length
)

function numberRate(camera: CameraStatus): number {
  return typeof camera.frame_rate === 'number'
    ? camera.frame_rate
    : (camera.frame_rate?.value ?? camera.fps)
}

function cameraError(camera: CameraStatus): string {
  return camera.last_error || (camera.source_errors
    ? `${camera.source_errors} capture source error${camera.source_errors === 1 ? '' : 's'}`
    : '')
}

function fromCamera(camera: CameraStatus): CameraDraft {
  return {
    camera_id: camera.camera_id,
    device_index: camera.device_index ?? 0,
    device_symbolic_link: camera.device_symbolic_link ?? '',
    width: camera.width,
    height: camera.height,
    frame_rate: numberRate(camera),
    format: camera.format ?? 'mjpeg',
    cuda_device: camera.cuda_device ?? 0,
    sample_queue_capacity: camera.sample_queue_capacity ?? 1,
    frame_pool_capacity: camera.frame_pool_capacity ?? 4,
    overflow: camera.overflow ?? 'drop-oldest',
    rotation: camera.rotation ?? 'none',
    allow_format_fallback: camera.allow_format_fallback ?? false,
    reconnect: camera.reconnect ?? true
  }
}

function keyFor(device: DiscoveredCamera): string {
  return device.device_symbolic_link || `index:${device.device_index}`
}

function configuredCamera(device: DiscoveredCamera): CameraStatus | undefined {
  return cameras.value.find((camera) =>
    device.device_symbolic_link
      ? camera.device_symbolic_link === device.device_symbolic_link
      : camera.device_index === device.device_index
  )
}

function cameraName(camera: CameraStatus): string {
  return discovered.value.find((device) =>
    camera.device_symbolic_link
      ? device.device_symbolic_link === camera.device_symbolic_link
      : device.device_index === camera.device_index
  )?.name || camera.device_symbolic_link || `Device ${camera.device_index ?? '—'}`
}

function rotationFor(device: DiscoveredCamera): CameraRotation {
  const key = keyFor(device)
  return rotationSelections.value[key] ?? configuredCamera(device)?.rotation ?? 'none'
}

function nextCameraId(): number {
  const used = new Set(cameras.value.map((camera) => camera.camera_id))
  let id = 0
  while (used.has(id)) id += 1
  return id
}

function openAdd(): void {
  if (dirty.value && !window.confirm('Discard unsaved camera changes?')) return
  mode.value = 'add'
  editingId.value = null
  draft.value = { ...blank(), camera_id: nextCameraId() }
  saved.value = { ...draft.value }
  error.value = ''
  notice.value = ''
}

function openEdit(camera: CameraStatus): void {
  if (dirty.value && !window.confirm('Discard unsaved camera changes?')) return
  mode.value = 'edit'
  editingId.value = camera.camera_id
  draft.value = fromCamera(camera)
  saved.value = fromCamera(camera)
  error.value = ''
  notice.value = ''
}

function cancel(): void {
  mode.value = 'closed'
  error.value = ''
  notice.value = ''
}

function validate(): string | null {
  if (!Number.isInteger(draft.value.camera_id) || draft.value.camera_id < 0)
    return 'Camera ID must be a non-negative integer.'
  if (mode.value === 'add' && cameras.value.some((camera) => camera.camera_id === draft.value.camera_id))
    return `Camera ID ${draft.value.camera_id} is already configured.`
  if (draft.value.device_index === null && !draft.value.device_symbolic_link.trim())
    return 'Choose a device index or symbolic link.'
  if (draft.value.device_index !== null && draft.value.device_index < 0)
    return 'Device index must be zero or greater.'
  if (draft.value.width < 1 || draft.value.height < 1) return 'Width and height must be positive.'
  if (draft.value.frame_rate <= 0) return 'Frame rate must be greater than 0 FPS.'
  if (draft.value.cuda_device < 0) return 'CUDA device must be zero or greater.'
  if (draft.value.sample_queue_capacity < 1 || draft.value.frame_pool_capacity < 1)
    return 'Queue capacities must be positive.'
  return null
}

function body(): Omit<CreateCameraRequest, 'camera_id'> {
  const { camera_id: _id, ...value } = draft.value
  return value
}

async function submit(): Promise<void> {
  const validation = validate()
  if (validation) {
    error.value = validation
    return
  }
  busy.value = true
  error.value = ''
  notice.value = mode.value === 'add' ? 'Enabling camera…' : 'Saving camera…'
  try {
    if (mode.value === 'add') {
      await api.addCamera({ camera_id: draft.value.camera_id, ...body() })
      editingId.value = draft.value.camera_id
    } else if (editingId.value !== null) {
      await api.configureCamera(editingId.value, body())
    }
    saved.value = { ...draft.value }
    mode.value = 'edit'
    notice.value = 'Camera settings applied'
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause)
    notice.value = ''
  } finally {
    busy.value = false
  }
}

async function scan(): Promise<void> {
  scanBusy.value = true
  scanError.value = ''
  try {
    discovered.value = await api.discoverCameras()
    scanComplete.value = true
  } catch (cause) {
    scanError.value = cause instanceof Error ? cause.message : String(cause)
  } finally {
    scanBusy.value = false
  }
}

async function enable(device: DiscoveredCamera): Promise<void> {
  const key = keyFor(device)
  if (configuredCamera(device) || enabledLocally.value[key] !== undefined) return
  addingKey.value = key
  scanError.value = ''
  try {
    const camera_id = nextCameraId()
    const rotation = rotationFor(device)
    await api.addCamera({
      camera_id,
      device_index: device.device_index,
      device_symbolic_link: device.device_symbolic_link,
      width: 1920,
      height: 1080,
      frame_rate: 30,
      format: 'mjpeg',
      cuda_device: 0,
      sample_queue_capacity: 1,
      frame_pool_capacity: 4,
      overflow: 'drop-oldest',
      rotation,
      allow_format_fallback: false,
      reconnect: true
    })
    enabledLocally.value = { ...enabledLocally.value, [key]: camera_id }
  } catch (cause) {
    scanError.value = cause instanceof Error ? cause.message : String(cause)
  } finally {
    addingKey.value = null
  }
}

async function setRotation(device: DiscoveredCamera, rotation: CameraRotation): Promise<void> {
  const key = keyFor(device)
  const previous = rotationFor(device)
  rotationSelections.value = { ...rotationSelections.value, [key]: rotation }
  rotationErrors.value = { ...rotationErrors.value, [key]: '' }
  const cameraId = configuredCamera(device)?.camera_id ?? enabledLocally.value[key]
  if (cameraId === undefined || rotation === previous) return
  rotationBusy.value = key
  try {
    await api.configureCamera(cameraId, { rotation })
  } catch (cause) {
    rotationSelections.value = { ...rotationSelections.value, [key]: previous }
    rotationErrors.value = {
      ...rotationErrors.value,
      [key]: cause instanceof Error ? cause.message : String(cause)
    }
  } finally {
    rotationBusy.value = null
  }
}

function rotationChanged(device: DiscoveredCamera, event: Event): void {
  if (!(event.target instanceof HTMLSelectElement)) return
  const value = event.target.value
  if (value === 'none' || value === 'cw90' || value === '180' || value === 'ccw90')
    void setRotation(device, value)
}

async function remove(camera: CameraStatus): Promise<void> {
  if (!window.confirm(`Remove Camera ${camera.camera_id}? This will stop capture for this camera and remove its configuration.`)) return
  busy.value = true
  error.value = ''
  notice.value = 'Removing camera…'
  try {
    await api.removeCamera(camera.camera_id)
    const next = { ...enabledLocally.value }
    for (const [key, id] of Object.entries(next)) if (id === camera.camera_id) delete next[key]
    enabledLocally.value = next
    if (editingId.value === camera.camera_id) cancel()
    notice.value = 'Camera removed'
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause)
    notice.value = ''
  } finally {
    busy.value = false
  }
}

function nextVideoCameraId(used: Set<number>): number {
  let id = 0
  while (used.has(id)) id += 1
  return id
}

async function chooseVideos(): Promise<void> {
  videoError.value = ''
  try {
    const paths = await api.pickVideoFiles()
    const used = new Set(videoFeeds.value.map((feed) => feed.camera_id))
    for (const path of paths) {
      const camera_id = nextVideoCameraId(used)
      used.add(camera_id)
      videoFeeds.value.push({ camera_id, path, rotation: 'none' })
    }
    videoFeeds.value = [...videoFeeds.value]
    if (paths.length) videoDirty.value = true
  } catch (cause) {
    videoError.value = cause instanceof Error ? cause.message : String(cause)
  }
}

function removeVideo(index: number): void {
  videoFeeds.value = videoFeeds.value.filter((_, itemIndex) => itemIndex !== index)
  videoDirty.value = true
}

function validateVideoFeeds(): string | null {
  if (!videoFeeds.value.length) return 'Select at least one video feed.'
  if (videoCudaDevice.value < 0 || videoFramePool.value < 1)
    return 'CUDA device must be zero or greater and frame pool capacity must be positive.'
  const ids = new Set<number>()
  for (const feed of videoFeeds.value) {
    if (!Number.isInteger(feed.camera_id) || feed.camera_id < 0)
      return 'Every camera ID must be a non-negative integer.'
    if (ids.has(feed.camera_id)) return `Camera ID ${feed.camera_id} is assigned more than once.`
    if (!feed.path.trim()) return `Choose a video file for Camera ${feed.camera_id}.`
    ids.add(feed.camera_id)
  }
  return null
}

async function submitVideoFeeds(): Promise<void> {
  const validation = validateVideoFeeds()
  if (validation) {
    videoError.value = validation
    return
  }
  videoBusy.value = true
  videoError.value = ''
  videoNotice.value = 'Submitting video feeds…'
  try {
    await api.request('/video-source', 'POST', {
      cameras: videoFeeds.value.map((feed) => ({ ...feed })),
      cuda_device: videoCudaDevice.value,
      frame_pool_capacity: videoFramePool.value,
      realtime: videoRealtime.value,
      loop: videoLoop.value
    })
    videoDirty.value = false
    sourcePanel.value = 'video'
    videoNotice.value = 'Video feeds applied'
  } catch (cause) {
    videoError.value = cause instanceof Error ? cause.message : String(cause)
    videoNotice.value = ''
  } finally {
    videoBusy.value = false
  }
}

async function useLiveCameras(): Promise<void> {
  videoBusy.value = true
  videoError.value = ''
  videoNotice.value = 'Switching to live cameras…'
  try {
    await api.request('/video-source', 'DELETE')
    sourcePanel.value = 'live'
    videoNotice.value = 'Live cameras selected'
  } catch (cause) {
    videoError.value = cause instanceof Error ? cause.message : String(cause)
    videoNotice.value = ''
  } finally {
    videoBusy.value = false
  }
}

watch(() => props.status.input_mode, (mode) => {
  if (mode === 'video') sourcePanel.value = 'video'
})
watch(() => props.status.video_inputs, (feeds) => {
  if (feeds && props.status.input_mode === 'video' && !videoDirty.value) {
    videoFeeds.value = feeds.map((feed) => ({ ...feed, rotation: feed.rotation ?? 'none' }))
    videoCudaDevice.value = props.status.video_cuda_device ?? 0
    videoFramePool.value = props.status.video_frame_pool_capacity ?? 8
    videoRealtime.value = props.status.video_realtime ?? false
    videoLoop.value = props.status.video_loop ?? false
  }
}, { immediate: true })

watch(() => props.status.cameras, (list) => {
  if (editingId.value !== null) {
    const current = list?.find((camera) => camera.camera_id === editingId.value)
    if (current && !dirty.value) {
      draft.value = fromCamera(current)
      saved.value = fromCamera(current)
    }
  }
  for (const device of discovered.value) {
    const camera = configuredCamera(device)
    const key = keyFor(device)
    if (camera?.rotation && rotationSelections.value[key] === camera.rotation) {
      const next = { ...rotationSelections.value }
      delete next[key]
      rotationSelections.value = next
    }
  }
})

onMounted(scan)
</script>

<template>
  <section class="camera-manager" aria-label="Camera manager">
    <header class="camera-manager-head">
      <div>
        <strong>FRAME SOURCES</strong>
        <small>Active: {{ props.status.input_mode === 'video' ? 'video files' : 'live cameras' }}</small>
      </div>
      <nav class="source-tabs" aria-label="Frame source configuration">
        <button :class="{ active: sourcePanel === 'live' }" @click="sourcePanel = 'live'">LIVE CAMERAS</button>
        <button :class="{ active: sourcePanel === 'video' }" @click="sourcePanel = 'video'">VIDEO FILES</button>
      </nav>
    </header>

    <section v-if="sourcePanel === 'video'" class="video-source-panel">
      <header class="camera-discovery-head">
        <div><strong>SYNCHRONIZED VIDEO FEEDS</strong><small>One frame from each file is submitted per batch, in lockstep.</small></div>
        <button class="camera-secondary-action" :disabled="videoBusy" @click="chooseVideos">ADD VIDEO FILES</button>
      </header>
      <p class="camera-empty">Use files captured by IRIS to replay the same frame indexes through the pipeline. All selected files must contain the same number of frames.</p>
      <label class="video-loop-toggle"><input v-model="videoLoop" type="checkbox" :disabled="videoBusy" @change="videoDirty = true" /><span><strong>LOOP FEEDS</strong><small>Restart all video files together when they reach the end.</small></span></label>
      <article v-for="(feed, index) in videoFeeds" :key="`${feed.camera_id}-${feed.path}`" class="video-feed-row">
        <label>Camera ID<input v-model.number="feed.camera_id" type="number" min="0" :disabled="videoBusy" @input="videoDirty = true" /></label>
        <div class="video-feed-path"><strong>FEED {{ String(feed.camera_id).padStart(2, '0') }}</strong><span :title="feed.path">{{ feed.path }}</span></div>
        <label>Rotation<select v-model="feed.rotation" :disabled="videoBusy" @change="videoDirty = true"><option value="none">None</option><option value="cw90">90° clockwise</option><option value="180">180°</option><option value="ccw90">90° counter-clockwise</option></select></label>
        <button class="danger-action" :disabled="videoBusy" @click="removeVideo(index)">REMOVE</button>
      </article>
      <div v-if="!videoFeeds.length" class="camera-empty">No video feeds selected.</div>
      <details class="video-options">
        <summary>Video processing options</summary>
        <div class="camera-field-row">
          <label>CUDA device<input v-model.number="videoCudaDevice" type="number" min="0" :disabled="videoBusy" /></label>
          <label>Frame pool capacity<input v-model.number="videoFramePool" type="number" min="1" :disabled="videoBusy" /></label>
        </div>
        <label><input v-model="videoRealtime" type="checkbox" :disabled="videoBusy" @change="videoDirty = true" /> Replay at source frame rate</label>
      </details>
      <p v-if="videoError" class="camera-form-error" role="alert">{{ videoError }}</p>
      <p v-if="videoNotice" class="camera-form-notice" role="status">{{ videoNotice }}</p>
      <div class="video-source-actions">
        <button class="primary-action" :disabled="videoBusy" @click="submitVideoFeeds">{{ videoBusy ? 'APPLYING…' : 'USE VIDEO FEEDS' }}</button>
        <button v-if="props.status.input_mode === 'video'" class="camera-secondary-action" :disabled="videoBusy" @click="useLiveCameras">RETURN TO LIVE CAMERAS</button>
      </div>
      <div v-if="props.status.input_mode === 'video'" class="active-video-feeds">
        <strong>ACTIVE VIDEO FEEDS</strong>
        <small>LOOP {{ props.status.video_loop ? 'ON' : 'OFF' }}</small>
        <small v-for="feed in props.status.video_inputs || []" :key="feed.camera_id">CAM_{{ String(feed.camera_id).padStart(2, '0') }} · ROTATION {{ feed.rotation ?? 'none' }} · {{ feed.path }}</small>
      </div>
    </section>

    <template v-else>
    <header class="camera-discovery-head">
      <div><strong>LIVE CAMERAS</strong><small>{{ cameras.length }} enabled · {{ connectedCount }} connected<span v-if="errorCount"> · {{ errorCount }} errors</span></small></div>
      <button class="camera-secondary-action" :disabled="busy || scanBusy || addingKey !== null" @click="openAdd">MANUAL</button>
    </header>
    <section class="camera-discovery" aria-labelledby="camera-discovery-heading">
      <header class="camera-discovery-head">
        <div>
          <strong id="camera-discovery-heading">AVAILABLE CAMERAS</strong>
          <small>{{ scanBusy ? 'Scanning connected devices…' : `${discovered.length} detected` }}</small>
        </div>
        <button class="camera-secondary-action" :disabled="scanBusy || busy || addingKey !== null" @click="scan">
          {{ scanBusy ? 'SCANNING…' : 'RESCAN' }}
        </button>
      </header>
      <p v-if="scanError" class="camera-form-error" role="alert">{{ scanError }}</p>
      <div v-else-if="scanBusy && !scanComplete" class="camera-empty">Looking for connected cameras…</div>
      <div v-else-if="scanComplete && !discovered.length" class="camera-empty">
        No cameras detected. Connect a camera and rescan.
      </div>
      <div v-else-if="!scanComplete" class="camera-empty">Camera discovery has not run yet.</div>
      <article v-for="device in discovered" :key="keyFor(device)" class="camera-device-card">
        <span :class="['camera-status-dot', configuredCamera(device) || enabledLocally[keyFor(device)] !== undefined ? 'connected' : 'idle']"></span>
        <div class="camera-device-info">
          <strong>{{ device.name || `Camera device ${device.device_index}` }}</strong>
          <small>Device {{ device.device_index }}<span v-if="configuredCamera(device)"> · Camera {{ configuredCamera(device)?.camera_id }} enabled</span></small>
          <small v-if="rotationErrors[keyFor(device)]" class="camera-error">{{ rotationErrors[keyFor(device)] }}</small>
        </div>
        <label class="camera-rotation-control">
          <span>Rotation</span>
          <select
            :value="rotationFor(device)"
            :disabled="busy || rotationBusy === keyFor(device) || addingKey === keyFor(device)"
            :aria-label="`Rotation for ${device.name}`"
            @change="rotationChanged(device, $event)"
          >
            <option value="none">0°</option>
            <option value="cw90">90° clockwise</option>
            <option value="180">180°</option>
            <option value="ccw90">90° counterclockwise</option>
          </select>
        </label>
        <button
          v-if="configuredCamera(device) || enabledLocally[keyFor(device)] !== undefined"
          class="camera-enabled-state"
          disabled
        >ENABLED</button>
        <button
          v-else
          class="primary-action camera-enable-action"
          :disabled="busy || addingKey !== null"
          @click="enable(device)"
        >
          {{ addingKey === keyFor(device) ? 'ENABLING…' : 'ENABLE' }}
        </button>
      </article>
    </section>

    <p v-if="notice && mode === 'closed'" class="camera-form-notice" role="status">{{ notice }}</p>
    <div v-if="!cameras.length" class="camera-empty">No cameras enabled yet. Enable a detected device to add it to capture.</div>
    <article v-for="camera in cameras" :key="camera.camera_id" class="camera-card">
      <div class="camera-card-main">
        <span :class="['camera-status-dot', camera.connected || camera.state === 'connected' ? 'connected' : cameraError(camera) ? 'error' : 'idle']"></span>
        <div>
          <strong>CAMERA {{ String(camera.camera_id).padStart(2, '0') }}</strong>
          <small>{{ cameraName(camera) }}</small>
          <small>{{ camera.width }} × {{ camera.height }} · {{ numberRate(camera) }} FPS · {{ camera.format || 'unknown' }}</small>
          <small v-if="cameraError(camera)" class="camera-error">{{ cameraError(camera) }}</small>
        </div>
      </div>
      <div class="camera-card-actions">
        <button :disabled="busy" @click="openEdit(camera)">EDIT</button>
        <button class="danger-action" :disabled="busy" @click="remove(camera)">REMOVE</button>
      </div>
    </article>

    <form v-if="mode !== 'closed'" class="camera-editor" @submit.prevent="submit">
      <header>
        <div>
          <strong>{{ mode === 'add' ? 'ADD CAMERA MANUALLY' : `EDIT CAMERA ${editingId}` }}</strong>
          <small v-if="dirty" class="unsaved">Unsaved changes</small>
        </div>
        <button type="button" :disabled="busy" @click="cancel">CANCEL</button>
      </header>
      <fieldset :disabled="busy">
        <legend>Identity</legend>
        <label>Camera ID<input v-model.number="draft.camera_id" type="number" min="0" /></label>
        <label>Connection method<select v-model="draft.device_index"><option :value="0">Device index</option><option :value="null">Symbolic link</option></select></label>
        <label v-if="draft.device_index !== null">Device index<input v-model.number="draft.device_index" type="number" min="0" /></label>
        <label v-else>Symbolic link<input v-model="draft.device_symbolic_link" placeholder="\\?\usb#…" /></label>
      </fieldset>
      <fieldset :disabled="busy">
        <legend>Capture</legend>
        <div class="camera-field-row">
          <label>Width (px)<input v-model.number="draft.width" type="number" min="1" /></label>
          <label>Height (px)<input v-model.number="draft.height" type="number" min="1" /></label>
        </div>
        <div class="camera-field-row">
          <label>Frame rate (FPS)<input v-model.number="draft.frame_rate" type="number" min="1" step="0.1" /></label>
          <label>Format<select v-model="draft.format"><option>mjpeg</option><option>yuy2</option><option>bgra8</option></select></label>
        </div>
        <label>Rotation<select v-model="draft.rotation"><option value="none">0°</option><option value="cw90">90° clockwise</option><option value="180">180°</option><option value="ccw90">90° counterclockwise</option></select></label>
        <label><input v-model="draft.allow_format_fallback" type="checkbox" /> Allow format fallback</label>
      </fieldset>
      <details>
        <summary>Advanced performance</summary>
        <fieldset :disabled="busy">
          <div class="camera-field-row">
            <label>CUDA device<input v-model.number="draft.cuda_device" type="number" min="0" /></label>
            <label>Sample queue<input v-model.number="draft.sample_queue_capacity" type="number" min="1" /></label>
          </div>
          <div class="camera-field-row">
            <label>Frame pool<input v-model.number="draft.frame_pool_capacity" type="number" min="1" /></label>
            <label>Overflow<select v-model="draft.overflow"><option>block</option><option>drop-oldest</option><option>drop-newest</option></select></label>
          </div>
          <label><input v-model="draft.reconnect" type="checkbox" /> Reconnect automatically</label>
        </fieldset>
      </details>
      <p v-if="error" class="camera-form-error" role="alert">{{ error }}</p>
      <p v-if="notice" class="camera-form-notice" role="status">{{ notice }}</p>
      <button class="primary-action" type="submit" :disabled="busy || (mode === 'edit' && !dirty)">
        {{ busy ? notice : mode === 'add' ? 'ENABLE CAMERA' : 'APPLY CHANGES' }}
      </button>
    </form>
    </template>
  </section>
</template>
