<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { CameraStatus, RuntimeStatus } from '../types/iris'
const props = defineProps<{ status: RuntimeStatus }>()
const playing = ref(true)
const selectedCamera = ref<number | null>(null)
const failedStreams = ref(new Set<number>())
const measuredFps = ref(new Map<number, number>())
const cameras = computed<CameraStatus[]>(() => props.status.cameras ?? [])
const previewPort = computed(() => props.status.preview?.port || 8080)
  const videoElements = new Map<number, HTMLVideoElement>()
let peer: RTCPeerConnection | undefined
let socket: WebSocket | undefined
let reconnectTimer: number | undefined
  let cameraOrder: number[] = []
  let reconnectDelay = 1000
  const fatalError = ref('')
  let connecting = false
  let connectionGeneration = 0
  let answerApplied = false
function markFailed(cameraId: number): void {
  failedStreams.value = new Set(failedStreams.value).add(cameraId)
}
function retry(cameraId: number): void {
  const next = new Set(failedStreams.value)
  next.delete(cameraId)
  failedStreams.value = next
  void connect()
}
function markLive(cameraId: number): void {
  if (!failedStreams.value.has(cameraId)) return
  const next = new Set(failedStreams.value)
  next.delete(cameraId)
  failedStreams.value = next
}
function fps(cameraId: number): number { return measuredFps.value.get(cameraId) ?? 0 }
function resetStreams(): void {
  measuredFps.value = new Map()
  void connect()
}
function setVideo(cameraId: number, element: Element | { $el?: Element } | null): void {
  if (element instanceof HTMLVideoElement) videoElements.set(cameraId, element)
  else if (element && '$el' in element && element.$el instanceof HTMLVideoElement) videoElements.set(cameraId, element.$el)
  else videoElements.delete(cameraId)
}
  function scheduleReconnect(): void {
    if (reconnectTimer || !playing.value || connecting || !cameras.value.length) return
    reconnectTimer = window.setTimeout(() => { reconnectTimer = undefined; void connect() }, reconnectDelay)
    reconnectDelay = Math.min(reconnectDelay * 2, 15000)
  }
  async function connect(): Promise<void> {
    if (connecting || !playing.value || !cameras.value.length) return
    connecting = true
    const generation = ++connectionGeneration
    answerApplied = false
    socket?.close(); peer?.close(); socket = undefined; peer = undefined
    fatalError.value = ''
    const ws = new WebSocket(`ws://127.0.0.1:${previewPort.value}/api/webrtc/signaling`)
  let connectionForSocket: RTCPeerConnection | undefined
  socket = ws
  ws.onopen = () => ws.send(JSON.stringify({
    type: 'hello',
    version: 1,
    cameras: cameras.value.map((camera) => ({ camera_id: camera.camera_id, width: camera.width, height: camera.height }))
  }))
    ws.onmessage = async (event) => {
      if (generation !== connectionGeneration || socket !== ws) return
      const message = JSON.parse(String(event.data)) as { type: string; cameras?: number[]; sdp?: string; message?: string }
      if (message.type === 'error') {
        fatalError.value = message.message ?? 'WebRTC negotiation failed'
        cameras.value.forEach((camera) => markFailed(camera.camera_id))
        ws.close(); peer?.close(); connecting = false; scheduleReconnect(); return
      }
    if (message.type === 'tracks') {
      cameraOrder = message.cameras?.length ? message.cameras : cameras.value.map((camera) => camera.camera_id)
      const connection = new RTCPeerConnection({ bundlePolicy: 'max-bundle' }); connectionForSocket = connection; peer = connection
      connection.onicecandidate = (candidate) => {
        if (generation === connectionGeneration && peer === connection && candidate.candidate && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'ice', candidate: candidate.candidate.candidate, sdpMid: candidate.candidate.sdpMid, sdpMLineIndex: candidate.candidate.sdpMLineIndex }))
      }
        connection.ontrack = (track) => {
        const cameraId = cameraOrder[track.transceiver.mid === null ? 0 : Number(track.transceiver.mid)]
        const video = videoElements.get(cameraId)
        if (video) { video.srcObject = new MediaStream([track.track]); void video.play(); markLive(cameraId); reconnectDelay = 1000 }
      }
        connection.onconnectionstatechange = () => { if (connection.connectionState === 'failed' || connection.connectionState === 'disconnected') { connecting = false; scheduleReconnect() } }
      cameraOrder.forEach(() => connection.addTransceiver('video', { direction: 'recvonly' }))
      const offer = await connection.createOffer(); await connection.setLocalDescription(offer)
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'offer', sdp: offer.sdp }))
    } else if (message.type === 'answer' && peer && message.sdp && peer === connectionForSocket) {
      if (answerApplied || peer.signalingState !== 'have-local-offer') return
      try {
        await peer.setRemoteDescription({ type: 'answer', sdp: message.sdp })
        answerApplied = true
        connecting = false
      } catch (error) {
        fatalError.value = error instanceof Error ? error.message : 'Failed to apply WebRTC answer'
        peer.close(); peer = undefined
        ws.close(); connecting = false; scheduleReconnect()
      }
    }
  }
  ws.onerror = () => { connecting = false; scheduleReconnect() }
  ws.onclose = () => { connecting = false; if (socket === ws) scheduleReconnect() }
}
watch(() => props.status.previewPublished, (current, previous) => {
  if (typeof current === 'number' && typeof previous === 'number' && current < previous) resetStreams()
})
let retryTimer: number | undefined
let fpsTimer: number | undefined
  onMounted(() => {
  void connect()
  retryTimer = window.setInterval(() => {
    for (const camera of cameras.value) {
      if (failedStreams.value.has(camera.camera_id)) retry(camera.camera_id)
    }
  }, 2500)
  fpsTimer = window.setInterval(async () => {
    if (!peer) return
    const stats = await peer.getStats(); const next = new Map<number, number>()
    stats.forEach((value) => { if (value.type === 'inbound-rtp' && value.kind === 'video') { const cameraId = cameraOrder[Number(value.mid ?? 0)]; next.set(cameraId, Number(value.framesPerSecond ?? 0)); if (value.packetsLost > 0) markFailed(cameraId) } })
    measuredFps.value = next
  }, 1000)
})
onBeforeUnmount(() => {
  if (retryTimer) window.clearInterval(retryTimer)
  if (fpsTimer) window.clearInterval(fpsTimer)
  if (reconnectTimer) window.clearTimeout(reconnectTimer)
  socket?.close(); peer?.close()
})
</script>

<template>
  <div class="camera-page">
    <header class="camera-page-head">
      <div>
        <span class="eyebrow">SOURCE MONITOR</span>
        <h2>Camera streams</h2>
        <p>{{ fatalError || (cameras.length ? `${cameras.length} configured view${cameras.length === 1 ? '' : 's'} from the active capture rig.` : 'No cameras configured.') }}</p>
      </div>
      <button class="stream-button" @click="playing = !playing">
        {{ playing ? 'Ⅱ  PAUSE ALL STREAMS' : '▶  PLAY ALL STREAMS' }}
      </button>
    </header>
    <div v-if="cameras.length" class="camera-grid-large" :class="{ focused: selectedCamera !== null }">
      <article v-for="camera in cameras" :key="camera.camera_id" class="camera-large" :class="{ selected: selectedCamera === camera.camera_id }" @dblclick="selectedCamera = selectedCamera === camera.camera_id ? null : camera.camera_id">
        <div class="camera-feed">
          <span class="feed-label">CAM_{{ String(camera.camera_id).padStart(2, '0') }}</span>
          <video v-if="playing && !failedStreams.has(camera.camera_id)" :ref="(element) => setVideo(camera.camera_id, element)" autoplay muted playsinline @error="markFailed(camera.camera_id)" />
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
.camera-feed video {
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
