param(
    [Parameter(Mandatory = $true)]
    [string[]]$Command
)

# Import MSVC (VS Build Tools / Visual Studio) environment into the current
# session, then run the requested command. Cargo on Windows needs link.exe,
# INCLUDE and LIB from a Visual Studio environment (VS2026 '18' BuildTools,
# which cargo's vswhere detection does not discover automatically).
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    Write-Error "vswhere.exe not found - Visual Studio / Build Tools missing"
    exit 1
}
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
    Write-Error "vcvars64.bat not found under $vs"
    exit 1
}

$envDump = cmd /c "`"$vcvars`" >nul 2>&1 && set"
if ($LASTEXITCODE -ne 0) {
    Write-Error "vcvars64.bat failed to initialize"
    exit 1
}
foreach ($line in $envDump) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}

& $Command[0] $Command[1..($Command.Length - 1)]
exit $LASTEXITCODE