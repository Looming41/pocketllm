# Build the pinned llama.cpp + PocketLLM engine patch on Windows and copy the binaries
# into bin\windows-<arch>\. Needs Git, CMake and Visual Studio Build Tools (C++).
#
# usage: powershell -ExecutionPolicy Bypass -File scripts\build_llamacpp.ps1 [-Backend cpu|vulkan|cuda]
param([string]$Backend = "cpu")
$ErrorActionPreference = "Stop"

$Commit = if ($env:LLAMA_COMMIT) { $env:LLAMA_COMMIT } else { "caa8191df159f764fb0160e3d2fc6be008948aee" }
$Root = Split-Path -Parent $PSScriptRoot
$Src = if ($env:LLAMA_SRC) { $env:LLAMA_SRC } else { Join-Path $Root ".build\llama.cpp" }
$Arch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "arm64" } else { "x64" }
Write-Host "llama.cpp $Commit  backend=$Backend  target=windows-$Arch"

function Run($exe, [string[]]$argv) {
  & $exe @argv
  if ($LASTEXITCODE -ne 0) { throw "$exe $($argv -join ' ') failed ($LASTEXITCODE)" }
}

if (-not (Test-Path (Join-Path $Src ".git"))) {
  Run git @("clone", "-c", "core.autocrlf=false", "--filter=blob:none", "https://github.com/ggml-org/llama.cpp", $Src)
}
# LF line endings, or the engine patch will not apply
Run git @("-C", $Src, "config", "core.autocrlf", "false")
& git -C $Src fetch --depth 1 origin $Commit 2>$null
if ($LASTEXITCODE -ne 0) { Run git @("-C", $Src, "fetch", "origin") }
Run git @("-C", $Src, "checkout", "-q", "-f", $Commit)
Run git @("-C", $Src, "clean", "-fdq", "ggml/src", "ggml/include", "src")
foreach ($p in Get-ChildItem (Join-Path $Root "patches\llama.cpp-*.patch")) {
  Write-Host "applying $($p.Name)"
  Run git @("-C", $Src, "apply", "--whitespace=nowarn", $p.FullName)
}

$Build = Join-Path $Src "build-$Backend"
$flags = @("-S", $Src, "-B", $Build, "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF", "-DGGML_RPC=ON")
switch ($Backend) {
  "vulkan" { $flags += "-DGGML_VULKAN=ON" }
  "cuda"   { $flags += "-DGGML_CUDA=ON" }
}
Run cmake $flags
Run cmake @("--build", $Build, "--config", "Release", "-j", "$env:NUMBER_OF_PROCESSORS", "--target",
            "llama-server", "llama-cli", "llama-bench", "llama-gguf-split", "llama-quantize", "llama-perplexity", "ggml-rpc-server")

$Dest = Join-Path $Root "bin\windows-$Arch"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$out = Join-Path $Build "bin\Release"
if (-not (Test-Path $out)) { $out = Join-Path $Build "bin" }
Copy-Item (Join-Path $out "*.exe"), (Join-Path $out "*.dll") -Destination $Dest -Force
Set-Content -Path (Join-Path $Dest "VERSION") -Value "$Commit $Backend"
Write-Host "installed into $Dest"
