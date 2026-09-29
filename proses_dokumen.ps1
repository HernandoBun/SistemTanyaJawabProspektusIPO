param([string[]]$Documents, [switch]$Check)
$ErrorActionPreference = 'Stop'
if ($Check) {
    & "$PSScriptRoot/jalankan.ps1" -Mode check
} else {
    & "$PSScriptRoot/jalankan.ps1" -Mode ingest -Documents $Documents
}
exit $LASTEXITCODE
