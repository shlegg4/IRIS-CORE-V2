$ErrorActionPreference = 'Stop'

$version = '1.26.5'
$downloadRoot = Join-Path $PSScriptRoot '..\build\downloads'
$installRoot = Join-Path $PSScriptRoot '..\build\gstreamer-dev'
$msi = Join-Path $downloadRoot "gstreamer-1.0-devel-msvc-x86_64-$version.msi"
$url = "https://gstreamer.freedesktop.org/data/pkg/windows/$version/msvc/gstreamer-1.0-devel-msvc-x86_64-$version.msi"
$devRoot = Join-Path $installRoot 'PFiles64\gstreamer\1.0\msvc_x86_64'
$downloadRoot = [System.IO.Path]::GetFullPath($downloadRoot)
$installRoot = [System.IO.Path]::GetFullPath($installRoot)
$msi = [System.IO.Path]::GetFullPath($msi)
$devRoot = [System.IO.Path]::GetFullPath($devRoot)

New-Item -ItemType Directory -Force -Path $downloadRoot, $installRoot | Out-Null
if (-not (Test-Path (Join-Path $devRoot 'include\gstreamer-1.0\gst\gst.h'))) {
    if (-not (Test-Path $msi)) {
        Write-Host "Downloading official GStreamer $version development package..."
        Add-Type -AssemblyName System.Net.Http
        $client = [System.Net.Http.HttpClient]::new()
        try {
            $response = $client.GetAsync($url, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
            $response.EnsureSuccessStatusCode()
            $total = $response.Content.Headers.ContentLength
            $input = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
            $output = [System.IO.File]::Create($msi)
            try {
                $buffer = New-Object byte[] 1048576
                [long]$downloaded = 0
                while (($read = $input.Read($buffer, 0, $buffer.Length)) -gt 0) {
                    $output.Write($buffer, 0, $read)
                    $downloaded += $read
                    if ($total) {
                        $percent = [math]::Floor(($downloaded * 100) / $total)
                        Write-Progress -Activity "Downloading GStreamer development package" -Status "$percent% ($([math]::Round($downloaded / 1MB, 1)) MB / $([math]::Round($total / 1MB, 1)) MB)" -PercentComplete $percent
                    }
                }
                Write-Progress -Activity "Downloading GStreamer development package" -Completed
            } finally { $output.Dispose(); $input.Dispose() }
        } finally { $client.Dispose() }
    }
    Write-Host 'Installing official GStreamer development package...'
    $logPath = Join-Path $downloadRoot 'gstreamer-dev-install.log'
    $arguments = @('/i', $msi, 'ADDLOCAL=ALL', '/qn', '/norestart', "INSTALLDIR=$installRoot", '/l*v', $logPath)
    $process = Start-Process msiexec.exe -ArgumentList $arguments -Verb RunAs -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        if ($process.ExitCode -eq 1625) {
            throw "GStreamer development MSI installation was blocked by Windows Installer policy (1625). Ask an administrator to allow this MSI. Installer log: $logPath"
        }
        throw "GStreamer development MSI installation failed with exit code $($process.ExitCode). Installer log: $logPath"
    }
}

$installedRoot = 'C:\Program Files\gstreamer\1.0\msvc_x86_64'
if (-not (Test-Path (Join-Path $devRoot 'include\gstreamer-1.0\gst\gst.h')) -and
    (Test-Path (Join-Path $installedRoot 'include\gstreamer-1.0\gst\gst.h'))) {
    Write-Host "Staging installed GStreamer development files in $installRoot..."
    New-Item -ItemType Directory -Force -Path $devRoot | Out-Null
    Get-ChildItem -LiteralPath $installedRoot -Force | Copy-Item -Destination $devRoot -Recurse -Force
}

if (-not (Test-Path (Join-Path $devRoot 'include\gstreamer-1.0\gst\gst.h'))) {
    throw "Official GStreamer development headers were not found under $devRoot"
}
if (-not (Get-ChildItem (Join-Path $devRoot 'lib') -Filter '*.lib' -ErrorAction SilentlyContinue)) {
    throw "Official GStreamer MSVC import libraries were not found under $(Join-Path $devRoot 'lib')"
}
Write-Host "IRIS_GSTREAMER_DEV_ROOT=$((Resolve-Path $devRoot).Path)"
