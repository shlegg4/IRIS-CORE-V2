param(
    [string]$TensorRtRoot = "C:\TensorRT-10.10.0.31",
    [string]$CacheRoot = (Join-Path $env:LOCALAPPDATA "IRIS\TensorRT"),
    [string]$DeploymentDirectory,
    [int]$Device = 0,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$cacheBase = [System.IO.Path]::GetFullPath($CacheRoot)
$trtexec = Join-Path $TensorRtRoot "bin\trtexec.exe"
$trtBin = Join-Path $TensorRtRoot "bin"
$trtLib = Join-Path $TensorRtRoot "lib"
$rtmoOnnx = Join-Path $repoRoot "models\source\rtmo-s\rtmo_s_candidates.onnx"
$da3Onnx = Join-Path $repoRoot "models\source\da3-base\da3-base-mv4-504.onnx"
$nvidiaSmi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue

if (-not (Test-Path -LiteralPath $trtexec)) { throw "trtexec not found: $trtexec" }
if (-not (Test-Path -LiteralPath $rtmoOnnx)) { throw "RTMO ONNX source missing: $rtmoOnnx" }
if (-not (Test-Path -LiteralPath $da3Onnx)) { throw "DA3 ONNX source missing. Stage the fixed four-view 504x504 DA3 ONNX graph and its .onnx.data file under models/source/da3-base." }
if (-not $nvidiaSmi) { throw "nvidia-smi is required to identify the target GPU." }

$gpuOutput = & $nvidiaSmi.Source "--query-gpu=name,compute_cap" "--format=csv,noheader" "-i" $Device
$gpuExitCode = $LASTEXITCODE
$gpu = $gpuOutput | Select-Object -First 1
if ($gpuExitCode -ne 0 -or -not $gpu) { throw "Could not query GPU $Device." }
$gpuParts = $gpu -split ",", 2
$gpuName = $gpuParts[0].Trim()
$computeCapability = $gpuParts[1].Trim()
$trtInfoOutput = & $trtexec "--help" 2>&1
$trtExitCode = $LASTEXITCODE
$trtInfo = ($trtInfoOutput -join [Environment]::NewLine).Trim()
if ($trtExitCode -ne 0) { throw "Could not query TensorRT version from $trtexec." }
$trtVersion = [regex]::Match($trtInfo, "TensorRT v[0-9]+").Value
if (-not $trtVersion) { throw "Could not parse TensorRT version." }

$rtmoHash = (Get-FileHash -LiteralPath $rtmoOnnx -Algorithm SHA256).Hash.ToLowerInvariant()
$da3Hash = (Get-FileHash -LiteralPath $da3Onnx -Algorithm SHA256).Hash.ToLowerInvariant()
$safeGpu = $gpuName -replace "[^A-Za-z0-9._-]", "_"
$cacheName = "${safeGpu}_sm${computeCapability}_$($trtVersion.Replace(' ',''))"
$cacheRoot = Join-Path $cacheBase $cacheName
$metadataPath = Join-Path $cacheRoot "metadata.json"
$rtmoEngine = Join-Path $cacheRoot "rtmo_s.engine"
$da3Engine = Join-Path $cacheRoot "da3_base.trt"
$logsDirectory = Join-Path $cacheRoot "logs"
New-Item -ItemType Directory -Force -Path $cacheRoot, $logsDirectory | Out-Null

$metadata = $null
if (Test-Path -LiteralPath $metadataPath) {
    try { $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json }
    catch { $metadata = $null }
}
$cacheValid = $metadata -and
    $metadata.gpu_name -eq $gpuName -and
    $metadata.compute_capability -eq $computeCapability -and
    $metadata.tensorrt_version -eq $trtVersion -and
    $metadata.rtmo_onnx_sha256 -eq $rtmoHash -and
    $metadata.da3_onnx_sha256 -eq $da3Hash -and
    (Test-Path -LiteralPath $rtmoEngine) -and
    (Test-Path -LiteralPath $da3Engine) -and
    $metadata.rtmo_engine_sha256 -eq (Get-FileHash -LiteralPath $rtmoEngine -Algorithm SHA256).Hash.ToLowerInvariant() -and
    $metadata.da3_engine_sha256 -eq (Get-FileHash -LiteralPath $da3Engine -Algorithm SHA256).Hash.ToLowerInvariant()

if ($Force -or -not $cacheValid) {
    $env:PATH = "$trtBin;$trtLib;$env:PATH"
    $builds = @(
        @{
            Name = "rtmo_s"
            Onnx = $rtmoOnnx
            Engine = $rtmoEngine
            Log = Join-Path $logsDirectory "rtmo_s.log"
            Shapes = @("--minShapes=images:3x3x640x640", "--optShapes=images:3x3x640x640", "--maxShapes=images:3x3x640x640")
        },
        @{
            Name = "da3_base"
            Onnx = $da3Onnx
            Engine = $da3Engine
            Log = Join-Path $logsDirectory "da3_base.log"
            Shapes = @("--minShapes=images:1x4x3x504x504", "--optShapes=images:1x4x3x504x504", "--maxShapes=images:1x4x3x504x504")
        }
    )
    foreach ($build in $builds) {
        $arguments = @(
            "--onnx=$($build.Onnx)",
            "--saveEngine=$($build.Engine)",
            "--fp16",
            "--device=$Device"
        ) + $build.Shapes
        Push-Location $repoRoot
        try {
            & $trtexec @arguments *> $build.Log
            $exitCode = $LASTEXITCODE
        }
        finally { Pop-Location }
        if ($exitCode -ne 0) {
            throw "TensorRT failed building $($build.Name). See $($build.Log)"
        }
    }

    $metadata = [ordered]@{
        gpu_name = $gpuName
        compute_capability = $computeCapability
        tensorrt_version = $trtVersion
        rtmo_onnx_sha256 = $rtmoHash
        da3_onnx_sha256 = $da3Hash
        rtmo_engine_sha256 = (Get-FileHash -LiteralPath $rtmoEngine -Algorithm SHA256).Hash.ToLowerInvariant()
        da3_engine_sha256 = (Get-FileHash -LiteralPath $da3Engine -Algorithm SHA256).Hash.ToLowerInvariant()
        built_utc = [DateTime]::UtcNow.ToString("o")
    }
    $metadata | ConvertTo-Json | Set-Content -LiteralPath $metadataPath -Encoding utf8
}

if ($DeploymentDirectory) {
    $assetDirectory = Join-Path (Resolve-Path $DeploymentDirectory).Path "assets"
    if (-not (Test-Path -LiteralPath $assetDirectory)) { throw "Deployment assets directory not found: $assetDirectory" }
    Copy-Item -LiteralPath $rtmoEngine -Destination (Join-Path $assetDirectory "rtmo_s.engine") -Force
    Copy-Item -LiteralPath $da3Engine -Destination (Join-Path $assetDirectory "da3_base.trt") -Force
}

Set-Content -LiteralPath (Join-Path $cacheBase "current.txt") -Value $cacheName -NoNewline -Encoding ascii

Get-Item -LiteralPath $rtmoEngine, $da3Engine | Select-Object Name, Length, FullName
Write-Host "GPU: $gpuName (SM $computeCapability); $trtVersion"
Write-Host "Host engine cache: $cacheRoot"
