$ErrorActionPreference = "Stop"

$runtimeRoot = "D:\MindBridgeRuntime\opensearch-3.9.0"
$temporaryRoot = Join-Path $runtimeRoot "tmp-jdk21"
New-Item -ItemType Directory -Force -Path $temporaryRoot | Out-Null
$env:OPENSEARCH_JAVA_HOME = "D:\MindBridgeJava\jdk-21.0.13+9"
$env:OPENSEARCH_TMPDIR = $temporaryRoot
$env:TEMP = $temporaryRoot
$env:TMP = $temporaryRoot

& (Join-Path $runtimeRoot "bin\opensearch.bat")
exit $LASTEXITCODE
