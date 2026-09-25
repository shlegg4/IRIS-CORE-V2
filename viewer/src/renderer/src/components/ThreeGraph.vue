<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { CalibrationSnapshot, PoseFrame, PosePerson } from '../types/iris'

const props = defineProps<{ showGrid?: boolean; showCameras?: boolean; pose?: PoseFrame | null; calibration?: CalibrationSnapshot | null }>()
const canvas = ref<HTMLCanvasElement | null>(null)
let renderer: THREE.WebGLRenderer | undefined
let animationFrame = 0
const links = [[0,1],[0,2],[1,3],[2,4],[5,6],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]]
const colors = [0x74e5df, 0xffb86b, 0xbd93f9, 0x50fa7b, 0xff79c6, 0xf1fa8c]

// Calibration and triangulation use camera coordinates (Y down); Three.js uses Y up.
const toScene = (v: number[]): THREE.Vector3 => new THREE.Vector3(Number(v[0]) || 0, -(Number(v[1]) || 0), Number(v[2]) || 0)
const peopleInFrame = (): PosePerson[] => props.pose?.people ?? (props.pose ? [props.pose] : [])

onMounted(() => {
  if (!canvas.value) return
  const scene = new THREE.Scene()
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100)
  camera.position.set(2.8, 1.5, 3.5)
  renderer = new THREE.WebGLRenderer({ canvas: canvas.value, antialias: true, alpha: true })
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
  const controls = new OrbitControls(camera, canvas.value)
  controls.enableDamping = true
  controls.target.set(0, 0, 0)
  const grid = new THREE.GridHelper(4, 16, 0x2b666d, 0x163439)
  const poseGroup = new THREE.Group(), cameraGroup = new THREE.Group()
  scene.add(grid, poseGroup, cameraGroup)
  const rigCenter = new THREE.Vector3()
  let rigScale = 1

  const normalizePoint = (point: THREE.Vector3): THREE.Vector3 => point.sub(rigCenter).multiplyScalar(rigScale)
  const cameraCenter = (r: number[], t: number[]): THREE.Vector3 => toScene([
    -(r[0]*t[0]+r[3]*t[1]+r[6]*t[2]),
    -(r[1]*t[0]+r[4]*t[1]+r[7]*t[2]),
    -(r[2]*t[0]+r[5]*t[1]+r[8]*t[2])
  ])

  const disposeGroup = (group: THREE.Group): void => {
    for (const child of [...group.children]) {
      child.removeFromParent()
      const object = child as THREE.Mesh | THREE.LineSegments
      object.geometry?.dispose()
      const material = object.material
      if (Array.isArray(material)) material.forEach((item) => item.dispose())
      else material?.dispose()
    }
  }
  const updatePoses = (): void => {
    disposeGroup(poseGroup)
    peopleInFrame().forEach((person, personIndex) => {
      const incoming = person.joints ?? person.joints3d ?? person.joints_3d
      if (!incoming || incoming.length < 17) return
      const valid = person.valid ?? person.jointValid ?? incoming.map(() => true)
      const points = incoming.map((point) => normalizePoint(toScene(point))), color = colors[personIndex % colors.length]
      const linePoints = links.filter(([a,b]) => valid[a] !== false && valid[b] !== false).flatMap(([a,b]) => [points[a], points[b]])
      if (linePoints.length) poseGroup.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(linePoints), new THREE.LineBasicMaterial({ color })))
      points.forEach((point, index) => {
        if (valid[index] === false) return
        const joint = new THREE.Mesh(new THREE.SphereGeometry(0.025, 8, 8), new THREE.MeshBasicMaterial({ color }))
        joint.position.copy(point); poseGroup.add(joint)
      })
    })
  }
  const updateCameras = (): void => {
    disposeGroup(cameraGroup)
    const calibrated = (props.calibration?.cameras ?? []).flatMap((calibration) => {
      const r = calibration.R_w2c, t = calibration.t_w2c
      return r.length < 9 || t.length < 3 ? [] : [{ r, center: cameraCenter(r, t) }]
    })
    if (calibrated.length > 1) {
      const bounds = new THREE.Box3().setFromPoints(calibrated.map(({ center }) => center))
      bounds.getCenter(rigCenter)
      // Keep the world ground at Y=0; center only in the ground plane.
      rigCenter.y = 0
      const size = bounds.getSize(new THREE.Vector3())
      const span = Math.max(size.x, size.y, size.z)
      // Fit the rig around the ground grid while preserving ground height.
      rigScale = span > 1e-6 ? 2.8 / span : 1
    } else {
      rigCenter.set(0, 0, 0)
      rigScale = 1
    }
    for (const { r, center: sourceCenter } of calibrated) {
      const center = normalizePoint(sourceCenter.clone())
      const direction = toScene([r[6], r[7], r[8]]).normalize()
      const viewCamera = new THREE.PerspectiveCamera(45, 1, 0.05, 0.35)
      viewCamera.position.copy(center); viewCamera.lookAt(center.clone().add(direction)); viewCamera.updateMatrixWorld()
      const frustum = new THREE.CameraHelper(viewCamera)
      frustum.setColors(new THREE.Color(0xffb86b), new THREE.Color(0xff7b72), new THREE.Color(0xffd166), new THREE.Color(0xffb86b), new THREE.Color(0xffb86b))
      cameraGroup.add(frustum)
    }
    updatePoses()
  }
  updateCameras()
  const stopPoseWatch = watch(() => props.pose, updatePoses, { deep: true })
  const stopCalibrationWatch = watch(() => props.calibration, updateCameras, { deep: true })
  const resize = (): void => {
    if (!canvas.value || !renderer) return
    camera.aspect = canvas.value.clientWidth / canvas.value.clientHeight
    camera.updateProjectionMatrix(); renderer.setSize(canvas.value.clientWidth, canvas.value.clientHeight, false)
  }
  const observer = new ResizeObserver(resize); observer.observe(canvas.value); resize()
  const tick = (): void => {
    grid.visible = props.showGrid !== false; cameraGroup.visible = props.showCameras !== false
    controls.update(); renderer?.render(scene, camera); animationFrame = requestAnimationFrame(tick)
  }
  tick()
  onBeforeUnmount(() => {
    cancelAnimationFrame(animationFrame); observer.disconnect(); controls.dispose(); stopPoseWatch(); stopCalibrationWatch()
    disposeGroup(poseGroup); disposeGroup(cameraGroup); grid.geometry.dispose()
    const gridMaterial = grid.material
    if (Array.isArray(gridMaterial)) gridMaterial.forEach((item) => item.dispose()); else gridMaterial.dispose()
    renderer?.dispose()
  })
})
</script>

<template><canvas ref="canvas" class="three-canvas" aria-label="Interactive Three.js multi-person 3D pose graph" /></template>
