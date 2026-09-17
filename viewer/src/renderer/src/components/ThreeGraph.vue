<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { PoseFrame } from '../types/iris'
const props = defineProps<{ showGrid?: boolean; pose?: PoseFrame | null }>()
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
  const posePoints = (): Array<[number, number, number]> => {
    const incoming = props.pose?.joints ?? props.pose?.joints3d ?? props.pose?.joints_3d
    if (!incoming || incoming.length < 15) return points
    const raw = incoming.map(
      (joint) =>
        [Number(joint[0]) || 0, Number(joint[1]) || 0, Number(joint[2]) || 0] as [
          number,
          number,
          number
        ]
    )
    const valid = raw.filter(
      (_, index) =>
        props.pose?.valid?.[index] !== false && props.pose?.jointValid?.[index] !== false
    )
    if (!valid.length) return points
    const min = [0, 1, 2].map((axis) => Math.min(...valid.map((joint) => joint[axis])))
    const max = [0, 1, 2].map((axis) => Math.max(...valid.map((joint) => joint[axis])))
    const scale = 1.7 / Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2], 0.001)
    const centerX = (min[0] + max[0]) / 2
    const centerZ = (min[2] + max[2]) / 2
    return raw.map(([x, y, z]) => [
      (x - centerX) * scale,
      (y - min[1]) * scale + 0.1,
      (z - centerZ) * scale
    ])
  }
  const updateLines = (next: Array<[number, number, number]>): void => {
    geo.setFromPoints(
      links.flatMap(([a, b]) => [new THREE.Vector3(...next[a]), new THREE.Vector3(...next[b])])
    )
  }
  updateLines(posePoints())
  scene.add(new THREE.LineSegments(geo, mat))
  const jgeo = new THREE.SphereGeometry(0.035, 10, 10),
    jmat = new THREE.MeshBasicMaterial({ color: 0xd4ffff })
  const joints: THREE.Mesh[] = []
  posePoints().forEach((p) => {
    const j = new THREE.Mesh(jgeo, jmat)
    j.position.set(...p)
    scene.add(j)
    joints.push(j)
  })
  const stopPoseWatch = watch(
    () => props.pose,
    () => {
      const next = posePoints()
      updateLines(next)
      joints.forEach((joint, index) => joint.position.set(...next[index]))
    },
    { deep: true }
  )
  const resize = (): void => {
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
  const tick = (): void => {
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
    stopPoseWatch()
    renderer?.dispose()
    geo.dispose()
    jgeo.dispose()
    mat.dispose()
    jmat.dispose()
  })
})
</script>
<template>
  <canvas ref="canvas" class="three-canvas" aria-label="Interactive Three.js 3D pose graph" />
</template>
