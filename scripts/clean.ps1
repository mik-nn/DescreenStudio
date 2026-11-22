# Repository cleanup: derived artefacts only (Data/, Temp/, ml/ runs, target/, caches).
# ASCII only: PowerShell 5.1 reads UTF-8 without BOM as ANSI (repo rule).
#
# Default is a REPORT. Nothing is deleted without -Apply.
#
#   powershell -NoProfile -File scripts\clean.ps1
#   powershell -NoProfile -File scripts\clean.ps1 -Apply
#   powershell -NoProfile -File scripts\clean.ps1 -What cargo -Apply
#
# Categories (-What, default = temp,data,ml,pycache):
#   temp     Temp/ experiment outputs and old acceptance runs
#   data     Data/ datasets except the ones the current pipeline needs
#   ml       ml/Archiv and old ml/train runs (keeps the shipped models)
#   pycache  __pycache__ / .pytest_cache
#   cargo    stale cargo build profiles (keeps target/<triple>/release)
#
# Never touched (input data and the models the product loads):
#   Data/RealScan, Data/chain_native, Data/gt, Data/unsplash-*
#   ml/train/_chain_native, ml/train/_chain_native_band
#   Temp/accept_native/src (scan tiffs), Temp/accept_native/c53full (latest
#   acceptance), Temp/accept_native/panels_253 (latest panels)
#   everything tracked by git (this script only removes ignored/derived paths)

param(
    [ValidateSet('all', 'temp', 'data', 'ml', 'cargo', 'pycache')]
    [string[]]$What = @('all'),
    [switch]$Apply
)

$ErrorActionPreference = 'Continue'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path.TrimEnd('\')
if ($What -contains 'all') { $What = @('temp', 'data', 'ml', 'pycache') }

$KeepData = @('RealScan', 'chain_native', 'gt', '_manifests')
$KeepDataWildcard = @('unsplash-*')
$KeepMlTrain = @('_chain_native', '_chain_native_band')
$KeepAccept = @('src', 'c53full', 'panels_253')

function Get-Size([string]$p) {
    if (-not (Test-Path -LiteralPath $p)) { return [int64]0 }
    $sum = [int64]0
    $stack = New-Object System.Collections.Stack
    $stack.Push($p)
    while ($stack.Count -gt 0) {
        $d = $stack.Pop()
        try {
            foreach ($f in [System.IO.Directory]::EnumerateFiles($d)) {
                try { $sum += (New-Object System.IO.FileInfo $f).Length } catch { }
            }
            foreach ($s in [System.IO.Directory]::EnumerateDirectories($d)) { $stack.Push($s) }
        } catch { }
    }
    return $sum
}

function Remove-Safe([string]$p) {
    $full = [System.IO.Path]::GetFullPath($p).TrimEnd('\')
    if (-not $full.StartsWith($Root + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "refusing to delete outside the repository: $full"
    }
    $lp = '\\?\' + $full
    if (Test-Path -LiteralPath $full -PathType Container) {
        try { [System.IO.Directory]::Delete($lp, $true) }
        catch { & cmd /c rd /s /q "$lp" 2>$null }
    } elseif (Test-Path -LiteralPath $full) {
        try { [System.IO.File]::Delete($lp) } catch { }
    }
    if (Test-Path -LiteralPath $full) { throw "failed to delete: $full" }
}

function Save-Manifest([string]$dir) {
    # Keep the recipe (seed, counts, presets) when the pixels go: dataset.json
    # is a few hundred bytes and makes the deleted set reproducible.
    $src = Join-Path $dir 'dataset.json'
    if (-not (Test-Path -LiteralPath $src -PathType Leaf)) { return }
    $dstDir = Join-Path $Root 'Data\_manifests'
    if (-not (Test-Path -LiteralPath $dstDir)) { [void](New-Item -ItemType Directory -Path $dstDir -Force) }
    $dst = Join-Path $dstDir ((Split-Path -Leaf $dir) + '.json')
    if (-not (Test-Path -LiteralPath $dst)) { Copy-Item -LiteralPath $src -Destination $dst -Force }
}

function Test-KeepData([string]$name) {
    if ($KeepData -contains $name) { return $true }
    foreach ($w in $KeepDataWildcard) { if ($name -like $w) { return $true } }
    return $false
}

$targets = New-Object System.Collections.ArrayList

if ($What -contains 'temp') {
    $temp = Join-Path $Root 'Temp'
    $accept = Join-Path $temp 'accept_native'
    if (Test-Path -LiteralPath $temp) {
        foreach ($d in Get-ChildItem -LiteralPath $temp -Directory -Force) {
            if ($d.Name -eq 'accept_native') { continue }
            [void]$targets.Add([pscustomobject]@{ Tag = 'temp'; Path = $d.FullName })
        }
        # Loose scratch files at the Temp root (logs, dumped scans, panels).
        foreach ($f in Get-ChildItem -LiteralPath $temp -File -Force) {
            if ($f.Extension -in @('.log', '.tif', '.tiff', '.png', '.json')) {
                [void]$targets.Add([pscustomobject]@{ Tag = 'temp'; Path = $f.FullName })
            }
        }
    }
    if (Test-Path -LiteralPath $accept) {
        foreach ($d in Get-ChildItem -LiteralPath $accept -Directory -Force) {
            if ($KeepAccept -contains $d.Name) { continue }
            [void]$targets.Add([pscustomobject]@{ Tag = 'temp'; Path = $d.FullName })
        }
    }
}

if ($What -contains 'data') {
    $data = Join-Path $Root 'Data'
    if (Test-Path -LiteralPath $data) {
        foreach ($d in Get-ChildItem -LiteralPath $data -Directory -Force) {
            if (Test-KeepData $d.Name) { continue }
            [void]$targets.Add([pscustomobject]@{ Tag = 'data'; Path = $d.FullName })
        }
    }
}

if ($What -contains 'ml') {
    $archiv = Join-Path $Root 'ml\Archiv'
    if (Test-Path -LiteralPath $archiv) {
        [void]$targets.Add([pscustomobject]@{ Tag = 'ml'; Path = $archiv })
    }
    $train = Join-Path $Root 'ml\train'
    if (Test-Path -LiteralPath $train) {
        foreach ($d in Get-ChildItem -LiteralPath $train -Directory -Force) {
            if ($d.Name -like '_run*' -or $d.Name -like '_chain*') {
                if ($KeepMlTrain -contains $d.Name) { continue }
                [void]$targets.Add([pscustomobject]@{ Tag = 'ml'; Path = $d.FullName })
            }
        }
    }
}

if ($What -contains 'cargo') {
    # .cargo/config.toml sets [build] target = x86_64-pc-windows-msvc, so:
    #   target/<triple>/release   - the shipped binary (keep)
    #   target/<triple>/debug     - debug build of the app (drop)
    #   target/<triple>/doc       - rustdoc (drop)
    #   target/release            - HOST artefacts: build scripts + proc-macros
    #                               for the release build (keep, else ~1 min
    #                               host rebuild)
    #   target/debug              - host artefacts of debug builds (drop)
    $target = Join-Path $Root 'src-tauri\target'
    foreach ($rel in @('debug', 'x86_64-pc-windows-msvc\debug', 'x86_64-pc-windows-msvc\doc')) {
        $p = Join-Path $target $rel
        if (Test-Path -LiteralPath $p) {
            [void]$targets.Add([pscustomobject]@{ Tag = 'cargo'; Path = $p })
        }
    }
}

if ($What -contains 'pycache') {
    $stack = New-Object System.Collections.Stack
    $stack.Push($Root)
    while ($stack.Count -gt 0) {
        $d = $stack.Pop()
        foreach ($s in (Get-ChildItem -LiteralPath $d -Directory -Force -ErrorAction SilentlyContinue)) {
            if ($s.Name -eq '.git' -or $s.Name -eq 'node_modules' -or $s.Name -eq 'target') { continue }
            if ($s.Name -eq '__pycache__' -or $s.Name -eq '.pytest_cache' -or $s.Name -eq '.mypy_cache') {
                [void]$targets.Add([pscustomobject]@{ Tag = 'pycache'; Path = $s.FullName })
            } else {
                $stack.Push($s.FullName)
            }
        }
    }
}

if ($targets.Count -eq 0) {
    Write-Output 'nothing to clean'
    exit 0
}

$rows = foreach ($t in $targets) {
    [pscustomobject]@{
        Tag  = $t.Tag
        GB   = [math]::Round((Get-Size $t.Path) / 1GB, 2)
        Path = $t.Path.Substring($Root.Length + 1)
    }
}
$rows = $rows | Sort-Object GB -Descending
$total = ($rows | Measure-Object -Property GB -Sum).Sum

Write-Output ("{0,-8} {1,7}  {2}" -f 'kind', 'size', 'path')
$rows | ForEach-Object { Write-Output ("{0,-8} {1,7:N2}  {2}" -f $_.Tag, $_.GB, $_.Path) }
Write-Output ("{0,-8} {1,7:N2}  TOTAL" -f '', $total)

Write-Output ''
Write-Output 'kept:'
Write-Output ('  Data: ' + (($KeepData + $KeepDataWildcard) -join ', '))
Write-Output ('  ml/train: ' + ($KeepMlTrain -join ', '))
Write-Output ('  Temp/accept_native: ' + ($KeepAccept -join ', '))

if (-not $Apply) {
    Write-Output ''
    Write-Output 'report only - rerun with -Apply to delete the paths above'
    exit 0
}

$left = 0.0
foreach ($t in $targets) {
    if ($t.Tag -eq 'data') { Save-Manifest $t.Path }
    try {
        Remove-Safe $t.Path
    } catch {
        Write-Output ("FAILED  {0}: {1}" -f $t.Path, $_.Exception.Message)
    }
}
foreach ($t in $targets) { $left += (Get-Size $t.Path) / 1GB }
Write-Output ''
Write-Output ("freed {0:N2} GB" -f ($total - $left))
if ($What -notcontains 'cargo') {
    Write-Output 'tip: -What cargo -Apply drops stale cargo profiles (keeps the release build)'
}
