param(
    [ValidateSet('app', 'ingest', 'check', 'test', 'evaluation')]
    [string]$Mode = 'app',
    [string[]]$Documents,
    [ValidateSet('chunk', 'page')]
    [string]$RelevanceUnit = 'chunk'
)
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$bundledPython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$pythonPath = $null
foreach ($candidate in @($projectPython, $bundledPython)) {
    if (Test-Path -LiteralPath $candidate) {
        $version = & $candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($LASTEXITCODE -eq 0 -and $version -eq '3.12') {
            $pythonPath = $candidate
            break
        }
    }
}
if (-not $pythonPath) { throw 'Siapkan Python 3.12 di .venv. Lihat README.md.' }
if ($Mode -eq 'ingest' -and -not $Documents) { throw 'Gunakan -Documents "RANS_2026.pdf".' }
$previousPythonPath = $env:PYTHONPATH
Push-Location $PSScriptRoot
try {
    # An installed virtual environment must not inherit incompatible binary packages.
    $env:PYTHONPATH = if ($pythonPath -eq $bundledPython) { Join-Path $PSScriptRoot '.python-deps' } else { $null }
    switch ($Mode) {
        'app' { & $pythonPath -X utf8 -m streamlit run app.py }
        'ingest' { & $pythonPath -X utf8 -m scripts.ingest_documents --documents @Documents }
        'test' { & $pythonPath -X utf8 -m pytest tests -q -p no:cacheprovider }
        'evaluation' { & $pythonPath -X utf8 -m scripts.run_evaluation --relevance-unit $RelevanceUnit }
        'check' {
            & $pythonPath -X utf8 -c "import numpy, chromadb, dotenv, streamlit; from llama_parse import LlamaParse; from langchain_openai import ChatOpenAI; from src.services.pipeline import RAGPipeline; p=RAGPipeline(); print('OK: dependensi LlamaParse dan inisialisasi pipeline. Belum menguji layanan API.')"
        }
    }
    $commandExitCode = $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
exit $commandExitCode
