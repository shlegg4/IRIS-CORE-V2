<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { CameraStatus, RuntimeStatus } from '../types/iris'
const props = defineProps<{ status: RuntimeStatus }>()
const playing = ref(true)
const selectedCamera = ref<number | null>(null)
const failedStreams = ref(new Set<number>())
const streamNonce = ref(new Map<number, number>())
const frameTimes = ref(new Map<number, number[]>())
const measuredFps = ref(new Map<number, number>())
const cameras = computed<CameraStatus[]>(() => props.status.cameras ?? [])
const previewPort = computed(() => props.status.preview?.port || 8080)
const streamUrl = (camera: CameraStatus): string =>
  `http://127.0.0.1:${previewPort.value}/api/preview/${camera.camera_id}.mjpeg`
function markFailed(cameraId: number): void {
  failedStreams.value = new Set(failedStreams.value).add(cameraId)
}
function retry(cameraId: number): void {
  const next = new Set(failedStreams.value)
  next.delete(cameraId)
  failedStreams.value = next
  const nonces = new Map(streamNonce.value)
  nonces.set(cameraId, (nonces.get(cameraId) ?? 0) + 1)
  streamNonce.value = nonces
}
function markLive(cameraId: number): void {
  if (!failedStreams.value.has(cameraId)) return
  const next = new Set(failedStreams.value)
  next.delete(cameraId)
  failedStreams.value = next
}
function recordFrame(cameraId: number): void {
  const now = Date.now()
  const next = new Map(frameTimes.value)
  next.set(cameraId, [...(next.get(cameraId) ?? []), now].filter((time) => now - time <= 2000))
  frameTimes.value = next
}
function fps(cameraId: number): number { return measuredFps.value.get(cameraId) ?? 0 }
function resetStreams(): void {
  const next = new Map(streamNonce.value)
  for (const camera of cameras.value) next.set(camera.camera_id, (next.get(camera.camera_id) ?? 0) + 1)
  streamNonce.value = next
  frameTimes.value = new Map()
  measuredFps.value = new Map()
}
watch(() => props.status.previewPublished, (current, previous) => {
  if (typeof current === 'number' && typeof previous === 'number' && current < previous) resetStreams()
})
let retryTimer: number | undefined
let fpsTimer: number | undefined
onMounted(() => {
  retryTimer = window.setInterval(() => {
    for (const camera of cameras.value) {
      if (failedStreams.value.has(camera.camera_id)) retry(camera.camera_id)
    }
  }, 2500)
  fpsTimer = window.setInterval(() => {
    const now = Date.now()
    const next = new Map<number, number>()
    for (const camera of cameras.value) {
      const history = frameTimes.value.get(camera.camera_id) ?? []
      const recent = history.filter((time) => now - time <= 1000)
      next.set(camera.camera_id, recent.length)
      // A feed that was live but has gone silent is usually an MJPEG
      // connection left behind while the runtime recreated the preview server.
      if (playing.value && history.length > 3 && !recent.length && !failedStreams.value.has(camera.camera_id)) {
        markFailed(camera.camera_id)
        retry(camera.camera_id)
      }
    }
    measuredFps.value = next
  }, 1000)
})
onBeforeUnmount(() => {
  if (retryTimer) window.clearInterval(retryTimer)
  if (fpsTimer) window.clearInterval(fpsTimer)
})
</script>

<template>
  <div class="camera-page">
    <header class="camera-page-head">
      <div>
        <span class="eyebrow">SOURCE MONITOR</span>
        <h2>Camera streams</h2>
        <p>{{ cameras.length ? `${cameras.length} configured view${cameras.length === 1 ? '' : 's'} from the active capture rig.` : 'No cameras configured.' }}</p>
      </div>
      <button class="stream-button" @click="playing = !playing">
        {{ playing ? 'Ⅱ  PAUSE ALL STREAMS' : '▶  PLAY ALL STREAMS' }}
      </button>
    </header>
    <div v-if="cameras.length" class="camera-grid-large" :class="{ focused: selectedCamera !== null }">
      <article v-for="camera in cameras" :key="camera.camera_id" class="camera-large" :class="{ selected: selectedCamera === camera.camera_id }" @dblclick="selectedCamera = selectedCamera === camera.camera_id ? null : camera.camera_id">
        <div class="camera-feed">
          <span class="feed-label">CAM_{{ String(camera.camera_id).padStart(2, '0') }}</span>
          <img v-if="playing && !failedStreams.has(camera.camera_id)" :key="`${camera.camera_id}-${streamNonce.get(camera.camera_id) ?? 0}`" :src="streamUrl(camera)" :alt="`Live feed from camera ${camera.camera_id}`" @load="recordFrame(camera.camera_id); markLive(camera.camera_id)" @error="markFailed(camera.camera_id)" />
          <span v-if="!playing" class="feed-placeholder">Ⅱ</span>
          <div v-else-if="failedStreams.has(camera.camera_id)" class="feed-error"><strong>STREAM UNAVAILABLE</strong><button @click="retry(camera.camera_id)">↻ RETRY</button></div>
          <span class="feed-live" :class="{ paused: !playing }"
            ><i></i>{{ playing ? 'LIVE' : 'PAUSED' }}</span
          ><span class="feed-time">{{ camera.width }} × {{ camera.height }} · {{ fps(camera.camera_id) }} FPS</span>
        </div>
        <div class="camera-info">
          <strong>CAMERA {{ String(camera.camera_id).padStart(2, '0') }}</strong
          ><span>{{ camera.fps.toFixed(1) }} FPS · {{ camera.width }} × {{ camera.height }}</span><span :class="['camera-ok', { 'camera-warn': failedStreams.has(camera.camera_id) }]">● {{ failedStreams.has(camera.camera_id) ? 'RECONNECTING' : 'CONNECTED' }}</span>
        </div>
      </article>
    </div>
    <div v-else class="empty-state">Configure at least one capture camera to start previewing streams.</div>
  </div>
</template>

<style scoped>
.camera-page {
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
.camera-page-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 20px;
  padding-bottom: 24px;
  border-bottom: 1px solid #1e292d;
}
.eyebrow {
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
.stream-button {
  flex: 0 0 auto;
  padding: 9px 12px;
  border: 1px solid #24545a;
  border-radius: 3px;
  background: #132528;
  color: #70e3e0;
  font-size: 9px;
  letter-spacing: 0.08em;
  white-space: nowrap;
}
.camera-grid-large {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  margin-top: 22px;
  padding-bottom: 1px;
}
.camera-large {
  min-width: 0;
  overflow: hidden;
  border: 1px solid #1e292d;
  background: #10181b;
}
.camera-feed {
  min-height: 180px;
  aspect-ratio: 16 / 9;
  position: relative;
  display: grid;
  place-items: center;
  overflow: hidden;
  background: radial-gradient(ellipse, #245058, #0b1518);
}
.camera-feed::after {
  content: '';
  position: absolute;
  inset: 0;
  pointer-events: none;
  background: repeating-linear-gradient(0deg, transparent 0 9px, #8be7e422 10px 11px);
}
.camera-feed img {
  width: 100%;
  height: 100%;
  min-height: 180px;
  object-fit: cover;
  display: block;
}
.feed-error {
  display: grid;
  gap: 10px;
  justify-items: center;
  color: #d9925f;
  font-size: 10px;
  letter-spacing: 0.08em;
}
.feed-error button {
  padding: 7px 10px;
  border: 1px solid #704a35;
  background: #231b18;
  color: #e0a77e;
  font-size: 9px;
}
.feed-placeholder {
  color: #5fd4d2;
  font-size: clamp(50px, 8vw, 80px);
  opacity: 0.65;
}
.feed-label,
.feed-live,
.feed-time {
  position: absolute;
  z-index: 1;
  font-size: 9px;
}
.feed-label {
  top: 12px;
  left: 12px;
}
.feed-live {
  top: 12px;
  right: 12px;
  color: #79dbb0;
}
.feed-live.paused {
  color: #d9925f;
}
.feed-live i {
  width: 6px;
  height: 6px;
  display: inline-block;
  margin-right: 5px;
  border-radius: 50%;
  background: currentColor;
}
.feed-time {
  bottom: 12px;
  left: 12px;
  color: #9eb3b4;
}
.camera-info {
  min-width: 0;
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: center;
  gap: 14px;
  padding: 13px;
  font-size: 9px;
}
.camera-info strong,
.camera-info span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.camera-info strong {
  font-weight: 500;
}
.camera-info span {
  color: #718087;
}
.camera-info .camera-ok {
  color: #79dbb0;
}
.camera-large.selected {
  border-color: #70e3e0;
}
.camera-grid-large.focused .camera-large:not(.selected) {
  display: none;
}
.empty-state {
  margin-top: 22px;
  padding: 40px 20px;
  border: 1px dashed #2a3b3f;
  color: #718087;
  text-align: center;
}
.camera-info .camera-warn {
  color: #d9925f;
}
@media (max-width: 900px) {
  .camera-grid-large {
    grid-template-columns: 1fr;
  }
}
@media (max-width: 650px) {
  .camera-page {
    padding: 20px 18px 28px;
  }
  .camera-page-head {
    align-items: stretch;
    flex-direction: column;
  }
  .stream-button {
    align-self: flex-start;
  }
}
@media (max-width: 440px) {
  .camera-info {
    grid-template-columns: 1fr auto;
  }
  .camera-info > span:nth-child(2) {
    display: none;
  }
}
</style>
