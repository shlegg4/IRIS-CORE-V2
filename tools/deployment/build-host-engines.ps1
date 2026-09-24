param(
    [string]$TensorRtRoot = "C:\TensorRT-10.10.0.31",
    [string]$CacheRoot = (Join-Path $env:LOCALAPPDATA "IRIS\TensorRT"),
    [string]$DeploymentDirectory,
    [string]$ModelRepo = "shlegg4/iris-models",
    [string]$ModelRevision = $(if ($env:IRIS_MODEL_REVISION) { $env:IRIS_MODEL_REVISION } else { "main" }),
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
if (-not $nvidiaSmi) { throw "nvidia-smi is required to identify the target GPU." }

$modelFiles = @(
    "models/source/rtmo-s/rtmo_s_candidates.onnx",
    "models/source/da3-base/da3-base-mv4-504.onnx",
    "models/source/da3-base/da3-base-mv4-504.onnx.data"
)
$missingModelFiles = @($modelFiles | Where-Object { -not (Test-Path -LiteralPath (Join-Path $repoRoot $_)) })
if ($missingModelFiles.Count -gt 0) {
    $hf = Get-Command "hf" -ErrorAction SilentlyContinue
    if (-not $hf) {
        throw "Missing model source files and the Hugging Face CLI ('hf') is not on PATH. Install huggingface_hub and make its Scripts directory available on PATH, then retry. Authentication is only needed for private repos. Missing: $($missingModelFiles -join ', ')"
    }
    $downloadArguments = @("download", $ModelRepo) + $missingModelFiles + @(
        "--revision", $ModelRevision,
        "--local-dir", $repoRoot
    )
    Write-Host "Downloading missing model inputs from $ModelRepo@$ModelRevision"
    & $hf.Source @downloadArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Hugging Face download failed. For a private repo, authenticate with 'hf auth login'."
    }
}

foreach ($requiredModelFile in $modelFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $requiredModelFile))) {
        throw "Required model source file was not downloaded: $requiredModelFile"
    }
}

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
$crashDumpDirectory = Join-Path ([System.IO.Path]::GetFullPath($CacheRoot)) "crash-dumps"
$werDumpRegistryPath = "HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\trtexec.exe"
$werDumpConfig = Get-ItemProperty -LiteralPath $werDumpRegistryPath -ErrorAction SilentlyContinue
$aeDebugConfig = Get-ItemProperty -LiteralPath "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\AeDebug" -ErrorAction SilentlyContinue
$werDumpEnabled = $werDumpConfig -and
    $werDumpConfig.DumpType -eq 1 -and
    $werDumpConfig.DumpCount -ge 1 -and
    $werDumpConfig.DumpFolder -and
    [System.IO.Path]::GetFullPath([string]$werDumpConfig.DumpFolder).TrimEnd("\") -eq $crashDumpDirectory.TrimEnd("\")

$rtmoHash = (Get-FileHash -LiteralPath $rtmoOnnx -Algorithm SHA256).Hash.ToLowerInvariant()
$da3Hash = (Get-FileHash -LiteralPath $da3Onnx -Algorithm SHA256).Hash.ToLowerInvariant()
$rtmoMinBatch = 1
$rtmoOptBatch = 3
$rtmoMaxBatch = 10
$safeGpu = $gpuName -replace "[^A-Za-z0-9._-]", "_"
$cacheName = "${safeGpu}_sm${computeCapability}_$($trtVersion.Replace(' ',''))"
$cacheRoot = Join-Path $cacheBase $cacheName
$metadataPath = Join-Path $cacheRoot "metadata.json"
$rtmoEngine = Join-Path $cacheRoot "rtmo_s.engine"
$da3Engine = Join-Path $cacheRoot "da3_base.trt"
$logsDirectory = Join-Path $cacheRoot "logs"
New-Item -ItemType Directory -Force -Path $cacheRoot, $logsDirectory | Out-Null
$runId = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ")
$runDirectory = Join-Path $logsDirectory $runId
$crashDumpDirectory = Join-Path $cacheBase "crash-dumps"
New-Item -ItemType Directory -Force -Path $runDirectory | Out-Null
$runManifestPath = Join-Path $runDirectory "run.json"

$env:PATH = "$trtBin;$trtLib;$env:PATH"
$gpuDetails = & $nvidiaSmi.Source "-q" 2>&1
$gpuDetails | Set-Content -LiteralPath (Join-Path $runDirectory "nvidia-smi.txt") -Encoding utf8
$nvcc = Get-Command "nvcc.exe" -ErrorAction SilentlyContinue
if ($nvcc) {
    $nvccDetails = & $nvcc.Source "--version" 2>&1
    $nvccDetails | Set-Content -LiteralPath (Join-Path $runDirectory "nvcc-version.txt") -Encoding utf8
}

$runManifest = [ordered]@{
    run_id = $runId
    started_utc = [DateTime]::UtcNow.ToString("o")
    computer_name = $env:COMPUTERNAME
    powershell_version = $PSVersionTable.PSVersion.ToString()
    os_version = [Environment]::OSVersion.VersionString
    repo_root = $repoRoot
    tensorrt_root = [System.IO.Path]::GetFullPath($TensorRtRoot)
    trtexec = $trtexec
    cache_root = $cacheRoot
    crash_dump_directory = $crashDumpDirectory
    gpu_name = $gpuName
    compute_capability = $computeCapability
    tensorrt_version = $trtVersion
    device = $Device
    path = $env:PATH
    driver_and_gpu_snapshot = "nvidia-smi.txt"
    cuda_compiler_snapshot = if ($nvcc) { "nvcc-version.txt" } else { $null }
    wer_minidump_configured = [bool]$werDumpEnabled
    wer_dump_folder = if ($werDumpEnabled) { [string]$werDumpConfig.DumpFolder } else { $null }
    postmortem_debugger = if ($aeDebugConfig.Debugger) { [string]$aeDebugConfig.Debugger } else { $null }
    builds = @()
}
if (-not $werDumpEnabled) {
    Write-Warning "Windows Error Reporting minidumps are not configured for trtexec.exe. From an elevated PowerShell window, run: tools/deployment/configure-trtexec-dumps.ps1 -DumpFolder `"$crashDumpDirectory`""
}
if ($aeDebugConfig.Debugger) {
    Write-Warning "A postmortem debugger is registered: $($aeDebugConfig.Debugger). Windows may skip WER LocalDumps while automatic debugging is configured; if the debugger prompt appears, capture the dump there."
}

$metadata = $null
if (Test-Path -LiteralPath $metadataPath) {
    try { $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json }
    catch { $metadata = $null }
}
$commonCacheValid = $metadata -and
    $metadata.gpu_name -eq $gpuName -and
    $metadata.compute_capability -eq $computeCapability -and
    $metadata.tensorrt_version -eq $trtVersion
$rtmoCacheValid = $false
$da3CacheValid = $false
if ($commonCacheValid -and -not $Force) {
    if ($metadata.rtmo -and
        $metadata.rtmo.onnx_sha256 -eq $rtmoHash -and
        $metadata.rtmo.min_batch -eq $rtmoMinBatch -and
        $metadata.rtmo.opt_batch -eq $rtmoOptBatch -and
        $metadata.rtmo.max_batch -eq $rtmoMaxBatch -and
        (Test-Path -LiteralPath $rtmoEngine)) {
        $rtmoCacheValid = $metadata.rtmo.engine_sha256 -eq (Get-FileHash -LiteralPath $rtmoEngine -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    if ($metadata.da3 -and
        $metadata.da3.onnx_sha256 -eq $da3Hash -and
        (Test-Path -LiteralPath $da3Engine)) {
        $da3CacheValid = $metadata.da3.engine_sha256 -eq (Get-FileHash -LiteralPath $da3Engine -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$builds = @(
    @{
        Name = "rtmo_s"
        Onnx = $rtmoOnnx
        OnnxHash = $rtmoHash
        Engine = $rtmoEngine
        CacheValid = $rtmoCacheValid
        CacheState = if ($rtmoCacheValid) { $metadata.rtmo } else { $null }
        Shapes = @(
            "--minShapes=images:${rtmoMinBatch}x3x640x640",
            "--optShapes=images:${rtmoOptBatch}x3x640x640",
            "--maxShapes=images:${rtmoMaxBatch}x3x640x640"
        )
    },
    @{
        Name = "da3_base"
        Onnx = $da3Onnx
        OnnxHash = $da3Hash
        Engine = $da3Engine
        CacheValid = $da3CacheValid
        CacheState = if ($da3CacheValid) { $metadata.da3 } else { $null }
        Shapes = @("--minShapes=images:1x4x3x504x504", "--optShapes=images:1x4x3x504x504", "--maxShapes=images:1x4x3x504x504")
    }
)

function Save-CacheMetadata {
    $payload = [ordered]@{
        metadata_version = 2
        gpu_name = $gpuName
        compute_capability = $computeCapability
        tensorrt_version = $trtVersion
        rtmo = $builds[0].CacheState
        da3 = $builds[1].CacheState
        updated_utc = [DateTime]::UtcNow.ToString("o")
    }
    $temporaryMetadataPath = "$metadataPath.$PID.tmp"
    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporaryMetadataPath -Encoding utf8
    Move-Item -LiteralPath $temporaryMetadataPath -Destination $metadataPath -Force
}

function Save-RunManifest {
    $runManifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $runManifestPath -Encoding utf8
}

foreach ($build in $builds) {
    if ($build.CacheValid) {
        Write-Host "Using cached $($build.Name) engine."
        $runManifest.builds += [ordered]@{ name = $build.Name; status = "cache_hit"; engine = $build.Engine }
        Save-RunManifest
        continue
    }

    $buildLog = Join-Path $runDirectory "$($build.Name).log"
    $temporaryEngine = Join-Path $cacheRoot "$($build.Name).$runId.building"
    $arguments = @(
        "--onnx=$($build.Onnx)",
        "--saveEngine=$temporaryEngine",
        "--fp16",
        "--device=$Device"
    ) + $build.Shapes
    $buildResult = [ordered]@{
        name = $build.Name
        status = "running"
        started_utc = [DateTime]::UtcNow.ToString("o")
        command = ('"{0}" {1}' -f $trtexec, ($arguments -join ' '))
        arguments = $arguments
        onnx = $build.Onnx
        onnx_sha256 = $build.OnnxHash
        engine = $build.Engine
        temporary_engine = $temporaryEngine
        log = $buildLog
    }
    $runManifest.builds += $buildResult
    Save-RunManifest

    Write-Host "Building $($build.Name). Log: $buildLog"
    Push-Location $repoRoot
    try {
        & $trtexec @arguments *> $buildLog
        $exitCode = [long]$LASTEXITCODE
    }
    finally { Pop-Location }

    $exitCodeUnsigned = $exitCode -band 0xFFFFFFFFL
    $exitCodeHex = '0x{0:X8}' -f $exitCodeUnsigned
    $buildResult.exit_code = $exitCode
    $buildResult.exit_code_hex = $exitCodeHex
    $buildResult.finished_utc = [DateTime]::UtcNow.ToString("o")
    if ($exitCode -ne 0) {
        $buildResult.status = "failed"
        Save-RunManifest
        throw "TensorRT failed building $($build.Name) (exit $exitCode, $exitCodeHex). See log: $buildLog. Windows crash dumps, if enabled, are in $cacheBase\crash-dumps. Partial engine: $temporaryEngine"
    }
    if (-not (Test-Path -LiteralPath $temporaryEngine) -or (Get-Item -LiteralPath $temporaryEngine).Length -le 0) {
        $buildResult.status = "failed"
        $buildResult.failure = "trtexec returned success but did not create a non-empty engine file."
        Save-RunManifest
        throw "TensorRT reported success building $($build.Name) but did not create an engine. See log: $buildLog"
    }

    Move-Item -LiteralPath $temporaryEngine -Destination $build.Engine -Force
    $engineHash = (Get-FileHash -LiteralPath $build.Engine -Algorithm SHA256).Hash.ToLowerInvariant()
    $state = [ordered]@{
        onnx_sha256 = $build.OnnxHash
        engine_sha256 = $engineHash
        built_utc = [DateTime]::UtcNow.ToString("o")
    }
    if ($build.Name -eq "rtmo_s") {
        $state.min_batch = $rtmoMinBatch
        $state.opt_batch = $rtmoOptBatch
        $state.max_batch = $rtmoMaxBatch
    }
    $build.CacheState = $state
    $buildResult.status = "built"
    $buildResult.engine_sha256 = $engineHash
    Save-CacheMetadata
    Save-RunManifest
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
Write-Host "Build diagnostics: $runDirectory"
Write-Host "Crash dumps (when WER is enabled): $crashDumpDirectory"
$runManifest.finished_utc = [DateTime]::UtcNow.ToString("o")
$runManifest.status = "succeeded"
Save-RunManifest
