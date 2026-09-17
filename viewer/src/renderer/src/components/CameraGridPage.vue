<script setup lang="ts">
import { ref } from 'vue'
const playing = ref(true)
</script>

<template>
  <div class="camera-page">
    <header class="camera-page-head">
      <div>
        <span class="eyebrow">SOURCE MONITOR</span>
        <h2>Camera streams</h2>
        <p>Four synchronized views from the active capture rig.</p>
      </div>
      <button class="stream-button" @click="playing = !playing">
        {{ playing ? 'Ⅱ  PAUSE ALL STREAMS' : '▶  PLAY ALL STREAMS' }}
      </button>
    </header>
    <div class="camera-grid-large">
      <article v-for="n in 4" :key="n" class="camera-large">
        <div class="camera-feed">
          <span class="feed-label">CAM_0{{ n }}</span
          ><span class="feed-placeholder">⌁</span
          ><span class="feed-live" :class="{ paused: !playing }"
            ><i></i>{{ playing ? 'LIVE' : 'PAUSED' }}</span
          ><span class="feed-time">00:00:14.28</span>
        </div>
        <div class="camera-info">
          <strong>CAMERA 0{{ n }}</strong
          ><span>{{ 59 + n }} FPS · 1280 × 720</span><span class="camera-ok">● CONNECTED</span>
        </div>
      </article>
    </div>
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
