$ErrorActionPreference = "Stop"

$assetRoot = "D:\Web agent\project_01"
$ollamaExe = Join-Path $assetRoot "Ollama\ollama.exe"
$ollamaModels = Join-Path $assetRoot "OllamaModels"
$ollamaHome = Join-Path $assetRoot "OllamaHome"
$ollamaAppData = Join-Path $assetRoot "OllamaAppData"

if (-not (Test-Path -LiteralPath $ollamaExe)) {
    throw "Ollama executable not found: $ollamaExe"
}

New-Item -ItemType Directory -Force -Path $ollamaModels, $ollamaHome, $ollamaAppData | Out-Null

# Keep every Ollama-created file on D: for this process.
$env:OLLAMA_MODELS = $ollamaModels
$env:USERPROFILE = $ollamaHome
$env:HOME = $ollamaHome
$env:LOCALAPPDATA = $ollamaAppData
$env:OLLAMA_HOST = "127.0.0.1:11434"

Write-Host "Ollama executable: $ollamaExe"
Write-Host "Ollama models:     $ollamaModels"
Write-Host "Ollama home:       $ollamaHome"
Write-Host "Ollama app data:   $ollamaAppData"

& $ollamaExe serve
