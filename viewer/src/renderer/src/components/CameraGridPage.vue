<script setup lang="ts">
import { ref } from 'vue'
const playing = ref(true)
</script>
<template>
  <div class="camera-page"><div class="camera-page-head"><div><span class="eyebrow">SOURCE MONITOR</span><h2>Camera streams</h2><p>Four synchronized views from the active capture rig.</p></div><button class="wide-button" @click="playing = !playing">{{ playing ? 'Ⅱ  PAUSE ALL STREAMS' : '▶  PLAY ALL STREAMS' }}</button></div><div class="camera-grid-large"><article v-for="n in 4" :key="n" class="camera-large"><div class="camera-feed"><span class="feed-label">CAM_0{{ n }}</span><span class="feed-placeholder">⌁</span><span class="feed-live"><i></i>{{ playing ? 'LIVE' : 'PAUSED' }}</span><span class="feed-time">00:00:14.28</span></div><div class="camera-info"><strong>CAMERA 0{{ n }}</strong><span>{{ 59 + n }} FPS · 1280 × 720</span><span class="camera-ok">● CONNECTED</span></div></article></div></div>
</template>
<style scoped>
.wide-button{background:#132528;border:1px solid #24545a;color:#70e3e0;border-radius:3px;padding:9px;font:inherit;font-size:9px;letter-spacing:.08em}
</style>
