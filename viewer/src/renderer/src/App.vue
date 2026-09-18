<script setup lang="ts">
import { computed, nextTick, onMounted, onBeforeUnmount, ref, watch } from 'vue'
import ThreeGraph from './components/ThreeGraph.vue'
import MetricsPage from './components/MetricsPage.vue'
import CameraGridPage from './components/CameraGridPage.vue'
import type { CalibrationSnapshot, MetricsSnapshot, PoseFrame, RuntimeStatus } from './types/iris'
const showGrid = ref(true),
  command = ref(''),
  activeTab = ref<'output' | 'metrics' | 'cameras'>('output')
const commandHistory = ref<string[]>([])
const historyIndex = ref(-1)
let historyDraft = ''
const logs = ref([
  ['10:42:01', 'system', 'IRIS runtime connected on localhost:8080'],
  ['10:42:02', 'ok', 'Loaded 4 camera streams · 60 FPS'],
  ['10:42:04', 'info', 'Pose pipeline initialized · RTMO-S']
])
const metrics = ref<MetricsSnapshot | null>(null)
const poseFrame = ref<PoseFrame | null>(null)
const calibration = ref<CalibrationSnapshot | null>(null)
const runtimeStatus = ref<RuntimeStatus>({})
const showCameras = ref(true)
const terminalOutput = ref<HTMLElement | null>(null)
const consoleWidth = ref(286)
const poseSequence = computed(
  () => poseFrame.value?.sequence ?? poseFrame.value?.sourceSequence ?? 0
)
const posePointCount = computed(
  () => poseFrame.value?.people?.reduce((total, person) =>
    total + (person.valid ?? person.jointValid ?? []).filter((valid) => valid).length, 0) ??
    (poseFrame.value?.joints ?? poseFrame.value?.joints3d ?? poseFrame.value?.joints_3d)?.length ?? 0
)
async function runCommand(): Promise<void> {
  const value = command.value.trim()
  if (value) {
    if (commandHistory.value.at(-1) !== value) commandHistory.value.push(value)
    historyIndex.value = -1
    historyDraft = ''
    logs.value.push([new Date().toLocaleTimeString('en-GB'), 'cmd', value])
    command.value = ''
    try {
      await window.api.sendCommand(value)
    } catch (error) {
      logs.value.push([
        new Date().toLocaleTimeString('en-GB'),
        'error',
        `Command was not sent: ${error instanceof Error ? error.message : String(error)}`
      ])
    }
  }
}
function handleCommandKeydown(event: KeyboardEvent): void {
  if (event.key === 'Tab') {
    event.preventDefault()
    const input = event.currentTarget as HTMLTextAreaElement
    const start = input.selectionStart
    const end = input.selectionEnd
    command.value = `${command.value.slice(0, start)}\t${command.value.slice(end)}`
    nextTick(() => input.setSelectionRange(start + 1, start + 1))
    return
  }

  if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return
  if (!commandHistory.value.length) return
  event.preventDefault()
  const input = event.currentTarget as HTMLTextAreaElement

  if (historyIndex.value === -1) historyDraft = command.value
  if (event.key === 'ArrowUp') {
    historyIndex.value = Math.min(historyIndex.value + 1, commandHistory.value.length - 1)
  } else if (historyIndex.value > 0) {
    historyIndex.value -= 1
  } else {
    historyIndex.value = -1
    command.value = historyDraft
    return
  }
  command.value = commandHistory.value[commandHistory.value.length - 1 - historyIndex.value]
  nextTick(() => {
    input.setSelectionRange(command.value.length, command.value.length)
  })
}
function startResize(event: PointerEvent): void {
  const startX = event.clientX
  const startWidth = consoleWidth.value
  const onMove = (move: PointerEvent): void => {
    const maximum = Math.max(220, window.innerWidth - 360)
    consoleWidth.value = Math.min(maximum, Math.max(220, startWidth + move.clientX - startX))
  }
  const onEnd = (): void => {
    window.removeEventListener('pointermove', onMove)
    window.removeEventListener('pointerup', onEnd)
  }
  window.addEventListener('pointermove', onMove)
  window.addEventListener('pointerup', onEnd, { once: true })
}
watch(
  logs,
  async (): Promise<void> => {
    await nextTick()
    if (terminalOutput.value) terminalOutput.value.scrollTop = terminalOutput.value.scrollHeight
  },
  { deep: true }
)
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
      runtimeStatus.value = value
      if (value.calibration) calibration.value = value.calibration
    })
  )
})
onBeforeUnmount(() => subscriptions.forEach((unsubscribe) => unsubscribe()))
</script>
<template>
  <main class="app-shell">
    <section class="workspace" :style="{ '--console-width': `${consoleWidth}px` }">
      <aside class="console panel">
        <div class="panel-heading">TERMINAL <span>⌁</span></div>
        <div ref="terminalOutput" class="terminal-output">
          <div v-for="(log, index) in logs" :key="index" class="log-line">
            <span class="time">{{ log[0] }}</span
            ><span :class="['kind', log[1]]">{{ log[1] }}</span
            ><span class="log-text">{{ log[2] }}</span>
          </div>
          <div class="cursor-line">
            <span class="prompt">›</span
            ><textarea
              v-model="command"
              autofocus
              placeholder="Enter command..."
              rows="1"
              spellcheck="false"
              @keydown="handleCommandKeydown"
              @keydown.enter.prevent="runCommand"
            ></textarea>
          </div>
        </div>
        <div class="console-footer">main · 4 workers <span>UTF-8</span></div>
      </aside>
      <div
        class="console-resizer"
        role="separator"
        aria-orientation="vertical"
        @pointerdown="startResize"
      ></div>
      <section class="main-stage">
        <div class="stage-toolbar">
          <div class="tabs">
            <button :class="{ active: activeTab === 'output' }" @click="activeTab = 'output'">
              3D OUTPUT</button
            ><button :class="{ active: activeTab === 'metrics' }" @click="activeTab = 'metrics'">
              METRICS</button
            ><button :class="{ active: activeTab === 'cameras' }" @click="activeTab = 'cameras'">
              CAMERA GRID
            </button>
          </div>
          <div v-if="activeTab === 'output'" class="stage-tools">
            <button @click="showGrid = !showGrid">{{ showGrid ? '▧ GRID' : '▧ GRID OFF' }}</button
            ><button @click="showCameras = !showCameras">{{ showCameras ? '◎ CAMERAS' : '◎ CAMERAS OFF' }}</button><button>⛶</button>
          </div>
        </div>
        <div v-if="activeTab === 'output'" class="viewport">
          <div class="viewport-label">
            <span class="pill green-pill"><span class="dot green"></span>LIVE</span
            ><span>FRAME {{ String(poseSequence).padStart(6, '0') }}</span
            ><span>1280 × 720</span>
          </div>
          <ThreeGraph :show-grid="showGrid" :show-cameras="showCameras" :pose="poseFrame" :calibration="calibration" />
          <div class="viewport-hud">
            <span>POSE_ESTIMATION</span><strong>RTMO-S</strong><span class="hud-divider"></span
            ><span
              >POINTS <strong>{{ posePointCount }}</strong></span
            >
          </div>
        </div>
        <MetricsPage v-else-if="activeTab === 'metrics'" :snapshot="metrics" />
        <CameraGridPage v-else :status="runtimeStatus" />
      </section>
    </section>
    <footer class="statusbar">
      <span><i class="dot green"></i> ALL SYSTEMS NOMINAL</span><span>GPU 38°C</span
      ><span>MEM 2.1 GB</span><span class="footer-right">IRIS RUNTIME v0.8.4 · LOCAL</span>
    </footer>
  </main>
</template>
