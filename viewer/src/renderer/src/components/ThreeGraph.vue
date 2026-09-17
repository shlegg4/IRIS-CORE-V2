<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
const props = defineProps<{ showGrid?: boolean }>()
const canvas = ref<HTMLCanvasElement | null>(null)
let renderer: THREE.WebGLRenderer | undefined
let frame = 0
const points: [[number, number, number], ...Array<[number, number, number]>] = [
  [0, 1.8, 0],
  [0, 1.55, 0],
  [0, 1.3, 0],
  [-0.35, 1.52, 0],
  [-0.65, 1.3, 0],
  [-0.8, 1.05, 0],
  [0.35, 1.52, 0],
  [0.65, 1.3, 0],
  [0.8, 1.05, 0],
  [-0.22, 1.08, 0],
  [-0.32, 0.58, 0],
  [-0.35, 0.1, 0],
  [0.22, 1.08, 0],
  [0.32, 0.58, 0],
  [0.35, 0.1, 0]
]
const links = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [4, 5],
  [2, 6],
  [6, 7],
  [7, 8],
  [2, 9],
  [9, 10],
  [10, 11],
  [2, 12],
  [12, 13],
  [13, 14]
]
onMounted(() => {
  if (!canvas.value) return
  const scene = new THREE.Scene(),
    camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100)
  camera.position.set(2.8, 1.5, 3.5)
  renderer = new THREE.WebGLRenderer({ canvas: canvas.value, antialias: true, alpha: true })
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
  const controls = new OrbitControls(camera, canvas.value)
  controls.enableDamping = true
  controls.target.set(0, 1, 0)
  const grid = new THREE.GridHelper(4, 16, 0x2b666d, 0x163439)
  scene.add(grid)
  const mat = new THREE.LineBasicMaterial({ color: 0x74e5df })
  const geo = new THREE.BufferGeometry()
  geo.setFromPoints(
    links.flatMap(([a, b]) => [new THREE.Vector3(...points[a]), new THREE.Vector3(...points[b])])
  )
  scene.add(new THREE.LineSegments(geo, mat))
  const jgeo = new THREE.SphereGeometry(0.035, 10, 10),
    jmat = new THREE.MeshBasicMaterial({ color: 0xd4ffff })
  points.forEach((p) => {
    const j = new THREE.Mesh(jgeo, jmat)
    j.position.set(...p)
    scene.add(j)
  })
  const resize = () => {
    if (!canvas.value || !renderer) return
    const w = canvas.value.clientWidth,
      h = canvas.value.clientHeight
    camera.aspect = w / h
    camera.updateProjectionMatrix()
    renderer.setSize(w, h, false)
  }
  const observer = new ResizeObserver(resize)
  observer.observe(canvas.value)
  resize()
  const tick = () => {
    grid.visible = props.showGrid !== false
    controls.update()
    renderer?.render(scene, camera)
    frame = requestAnimationFrame(tick)
  }
  tick()
  onBeforeUnmount(() => {
    cancelAnimationFrame(frame)
    observer.disconnect()
    controls.dispose()
    renderer?.dispose()
    geo.dispose()
    jgeo.dispose()
    mat.dispose()
    jmat.dispose()
  })
})
</script>
<template>
  <canvas
    ref="canvas"
    class="three-canvas"
    style="position: absolute; inset: 0; width: 100%; height: 100%; display: block; cursor: grab"
    aria-label="Interactive Three.js 3D pose graph"
  />
</template>
