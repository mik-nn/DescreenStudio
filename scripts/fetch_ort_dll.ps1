param(
    [string]$Dest = (Join-Path (Split-Path $PSScriptRoot -Parent) "src-tauri\deps")
)

# Copy onnxruntime.dll + DirectML.dll from the active Python wheel
# (onnxruntime-directml) into src-tauri\deps. Kept out of git (see .gitignore);
# required at runtime by `ort` crate's load-dynamic backend.
$env:PYTHONIOENCODING = "utf-8"
$src = python -c "import onnxruntime, pathlib; print(pathlib.Path(onnxruntime.__file__).parent / 'capi')"
if (-not $?) { Write-Error "Python onnxruntime not found" ; exit 1 }
if (-not (Test-Path -LiteralPath $Dest)) { New-Item -ItemType Directory -Path $Dest | Out-Null }
foreach ($dll in @("onnxruntime.dll", "DirectML.dll")) {
    $from = Join-Path $src $dll
    if (-not (Test-Path -LiteralPath $from)) { Write-Error "missing $from"; exit 1 }
    Copy-Item -LiteralPath $from -Destination (Join-Path $Dest $dll) -Force
    Write-Output "copied $dll -> $Dest"
}