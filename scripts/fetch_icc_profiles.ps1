# (Re)fetches the 4 ECI offset ICC profiles used by the halftone simulator
# (see ml/simulator/icc/manifest.json). The *.icc binaries are gitignored;
# this script restores them from the ECI Offset 2009 package (free download).
$ErrorActionPreference = 'Stop'
$zip = "$env:TEMP\opencode\eci_offset_2009.zip"
$dst = "F:\DescreenStudioPro\ml\simulator\icc"
$want = @(
    'ISOcoated_v2_eci.icc',
    'PSO_Uncoated_ISO12647_eci.icc',
    'ISOuncoatedyellowish.icc',
    'PSO_SNP_Paper_eci.icc'
)
if (-not (Test-Path -LiteralPath $zip)) {
    Invoke-WebRequest -Uri 'https://eci.org/lib/exe/eci_offset_2009.zip' -OutFile $zip
}
$tmp = Join-Path ([IO.Path]::GetTempPath()) 'opencode\eci2009_restored'
Expand-Archive -LiteralPath $zip -DestinationPath $tmp -Force
New-Item -ItemType Directory -Path $dst -Force | Out-Null
Get-ChildItem -LiteralPath $tmp -Recurse -Filter '*.icc' |
    Where-Object { $want -contains $_.Name } |
    Copy-Item -Destination $dst -Force
Get-ChildItem -LiteralPath $dst -Filter '*.icc' | Select-Object Name, Length
