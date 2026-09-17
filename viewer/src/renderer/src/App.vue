<script setup lang="ts">
import { computed, onMounted, onBeforeUnmount, ref } from 'vue'
import ThreeGraph from './components/ThreeGraph.vue'
import MetricsPage from './components/MetricsPage.vue'
import CameraGridPage from './components/CameraGridPage.vue'
import type { MetricsSnapshot, PoseFrame } from './types/iris'
const showGrid = ref(true),
  command = ref(''),
  activeTab = ref<'output' | 'metrics' | 'cameras'>('output')
const logs = ref([
  ['10:42:01', 'system', 'IRIS runtime connected on localhost:8080'],
  ['10:42:02', 'ok', 'Loaded 4 camera streams · 60 FPS'],
  ['10:42:04', 'info', 'Pose pipeline initialized · RTMO-S']
])
const metrics = ref<MetricsSnapshot | null>(null)
const poseFrame = ref<PoseFrame | null>(null)
const poseSequence = computed(
  () => poseFrame.value?.sequence ?? poseFrame.value?.sourceSequence ?? 0
)
const posePointCount = computed(
  () =>
    (poseFrame.value?.joints ?? poseFrame.value?.joints3d ?? poseFrame.value?.joints_3d)?.length ??
    0
)
function runCommand(): void {
  const value = command.value.trim()
  if (value) {
    logs.value.push(['10:42:12', 'cmd', value])
    void window.api.sendCommand(value)
    command.value = ''
  }
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
    })
  )
})
onBeforeUnmount(() => subscriptions.forEach((unsubscribe) => unsubscribe()))
</script>
<template>
  <main class="app-shell">
    <section class="workspace">
      <aside class="console panel">
        <div class="panel-heading">TERMINAL <span>⌁</span></div>
        <div class="terminal-output">
          <div v-for="(log, index) in logs" :key="index" class="log-line">
            <span class="time">{{ log[0] }}</span
            ><span :class="['kind', log[1]]">{{ log[1] }}</span
            ><span class="log-text">{{ log[2] }}</span>
          </div>
          <div class="cursor-line">
            <span class="prompt">›</span
            ><input
              v-model="command"
              autofocus
              placeholder="Enter command..."
              @keyup.enter="runCommand"
            />
          </div>
        </div>
        <div class="console-footer">main · 4 workers <span>UTF-8</span></div>
      </aside>
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
            ><button>⛶</button>
          </div>
        </div>
        <div v-if="activeTab === 'output'" class="viewport">
          <div class="viewport-label">
            <span class="pill green-pill"><span class="dot green"></span>LIVE</span
            ><span>FRAME {{ String(poseSequence).padStart(6, '0') }}</span
            ><span>1280 × 720</span>
          </div>
          <ThreeGraph :show-grid="showGrid" :pose="poseFrame" />
          <div class="viewport-hud">
            <span>POSE_ESTIMATION</span><strong>RTMO-S</strong><span class="hud-divider"></span
            ><span
              >POINTS <strong>{{ posePointCount }}</strong></span
            >
          </div>
        </div>
        <MetricsPage v-else-if="activeTab === 'metrics'" :snapshot="metrics" /><CameraGridPage
          v-else
        />
      </section>
    </section>
    <footer class="statusbar">
      <span><i class="dot green"></i> ALL SYSTEMS NOMINAL</span><span>GPU 38°C</span
      ><span>MEM 2.1 GB</span><span class="footer-right">IRIS RUNTIME v0.8.4 · LOCAL</span>
    </footer>
  </main>
</template>
