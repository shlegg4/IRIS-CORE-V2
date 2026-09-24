param(
    [switch]$Disable,
    [string]$DumpFolder
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $DumpFolder) { $DumpFolder = Join-Path $repoRoot "models\cache\crash-dumps" }
$dumpFolder = [System.IO.Path]::GetFullPath($DumpFolder)
$registryPath = "HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\trtexec.exe"
$expectedDumpFolder = [System.IO.Path]::GetFullPath($dumpFolder).TrimEnd("\")
$principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())

if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Windows requires administrator privileges to configure per-application WER dumps. Re-run this script from an elevated PowerShell window."
}

if ($Disable) {
    if (-not (Test-Path -LiteralPath $registryPath)) {
        Write-Host "No per-application trtexec.exe dump configuration is present."
        exit 0
    }

    $existing = Get-ItemProperty -LiteralPath $registryPath
    $existingDumpFolder = if ($existing.DumpFolder) { [System.IO.Path]::GetFullPath([string]$existing.DumpFolder).TrimEnd("\") } else { $null }
    if ($existingDumpFolder -ne $expectedDumpFolder -or $existing.DumpType -ne 1 -or $existing.DumpCount -ne 3) {
        throw "The trtexec.exe WER settings do not match this script's minidump configuration; leaving them unchanged."
    }

    Remove-ItemProperty -LiteralPath $registryPath -Name DumpFolder, DumpType, DumpCount
    Remove-Item -LiteralPath $registryPath -Force
    Write-Host "Removed this script's trtexec.exe WER minidump configuration."
    exit 0
}

if (Test-Path -LiteralPath $registryPath) {
    $existing = Get-ItemProperty -LiteralPath $registryPath
    $existingDumpFolder = if ($existing.DumpFolder) { [System.IO.Path]::GetFullPath([string]$existing.DumpFolder).TrimEnd("\") } else { $null }
    if (($existingDumpFolder -and $existingDumpFolder -ne $expectedDumpFolder) -or
        ($null -ne $existing.DumpType -and $existing.DumpType -ne 1) -or
        ($null -ne $existing.DumpCount -and $existing.DumpCount -ne 3)) {
        throw "A different trtexec.exe WER dump configuration already exists at $registryPath; inspect it before changing it."
    }
}

New-Item -ItemType Directory -Force -Path $dumpFolder | Out-Null
New-Item -Path $registryPath -Force | Out-Null
New-ItemProperty -LiteralPath $registryPath -Name DumpFolder -PropertyType ExpandString -Value $dumpFolder -Force | Out-Null
New-ItemProperty -LiteralPath $registryPath -Name DumpType -PropertyType DWord -Value 1 -Force | Out-Null
New-ItemProperty -LiteralPath $registryPath -Name DumpCount -PropertyType DWord -Value 3 -Force | Out-Null

Write-Host "Windows Error Reporting minidumps are enabled for trtexec.exe."
Write-Host "Dump folder: $dumpFolder"
Write-Host "Dump type: minidump; maximum retained: 3"
$aeDebugConfig = Get-ItemProperty -LiteralPath "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\AeDebug" -ErrorAction SilentlyContinue
if ($aeDebugConfig.Debugger) {
    Write-Warning "A postmortem debugger is registered: $($aeDebugConfig.Debugger). Windows may skip WER LocalDumps while automatic debugging is configured."
}
