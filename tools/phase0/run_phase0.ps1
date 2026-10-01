# Reproduces the Phase 0 measurements on a Windows machine with MinGW-w64 g++ and (optionally) rustc.
# Usage: powershell -ExecutionPolicy Bypass -File tools\phase0\run_phase0.ps1 [-OutDir build\phase0] [-Steps 300]
param([string]$OutDir = "build\phase0", [int]$Steps = 300)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$fx = Join-Path $root "external\FluidX3D"
$out = Join-Path $root $OutDir
New-Item -ItemType Directory -Force $out | Out-Null
$inc = Join-Path $fx "src\OpenCL\include"
$lib = Join-Path $fx "src\OpenCL\lib\OpenCL.lib"

function Build-Variant([string]$name, [string]$precision, [string[]]$patches, [string]$extra = "") {
	$dir = Join-Path $out "src_$name"
	if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
	Copy-Item -Recurse (Join-Path $fx "src") $dir
	$rel = $dir.Substring($root.Length + 1).Replace("\", "/") # git apply resolves paths from the repo root
	foreach ($p in $patches) { git -C $root apply -p2 --directory=$rel (Join-Path $PSScriptRoot "patches\$p"); if ($LASTEXITCODE) { throw "patch $p failed" } }
	Copy-Item (Join-Path $PSScriptRoot "bench_setup.cpp") (Join-Path $dir "setup.cpp") -Force
	$d = [IO.File]::ReadAllText((Join-Path $dir "defines.hpp")).Replace("#define FP16S //", "//#define FP16S //")
	if ($precision -eq "fp16s") { $d = $d.Replace("//#define FP16S //", "#define FP16S //") }
	if ($precision -eq "fp16c") { $d = $d.Replace("//#define FP16C //", "#define FP16C //") }
	[IO.File]::WriteAllText((Join-Path $dir "defines.hpp"), $d)
	$exe = Join-Path $out "bench_$name.exe"
	cmd /c "g++ `"$dir\*.cpp`" -o `"$exe`" -std=c++17 -pthread -O3 -Wno-comment $extra -I`"$inc`" `"$lib`""
	if ($LASTEXITCODE) { throw "build $name failed" }
	return $exe
}

function Run-Bench([string]$exe, [string]$sizes, [hashtable]$envs = @{}) {
	$env:NS_SIZES = $sizes; $env:NS_STEPS = "$Steps"; $env:NS_DX = "1"; $env:NS_PAD = "0"
	foreach ($k in $envs.Keys) { Set-Item "env:$k" $envs[$k] }
	$log = [IO.Path]::GetTempFileName()
	Start-Process -FilePath $exe -WorkingDirectory $out -RedirectStandardOutput "$log.out" -RedirectStandardError $log -NoNewWindow -Wait | Out-Null
	Get-Content $log | Where-Object { $_ -match "^RESULT" }
}

$sweep = "32,48,64,96,128,160,192,256"
$results = @{}
$results["sweep_fp32"]   = Run-Bench (Build-Variant "fp32"  "fp32"  @()) $sweep
$results["sweep_fp16s"]  = Run-Bench (Build-Variant "fp16s" "fp16s" @()) $sweep
$results["sweep_fp16c"]  = Run-Bench (Build-Variant "fp16c" "fp16c" @()) $sweep
$results["sweep_fp16s_nosync"] = Run-Bench (Build-Variant "fp16s_nosync" "fp16s" @("0001-optional-per-step-sync.patch") "-DNS_NO_STEP_SYNC") $sweep
$pad = Build-Variant "fp16s_pad" "fp16s" @("0002-ddf-stride-padding.patch")
foreach ($p in 0, 64, 256, 2112) { $results["pad_$p"] = Run-Bench $pad "128,256,136" @{ NS_PAD = "$p" } }
$fp16s = Join-Path $out "bench_fp16s.exe"
$results["domains_dx1"] = Run-Bench $fp16s "64,128,160" @{ NS_DX = "1" }
$results["domains_dx2"] = Run-Bench $fp16s "64,128,160" @{ NS_DX = "2" }

cmd /c "g++ `"$PSScriptRoot\bwprobe.cpp`" -o `"$out\bwprobe.exe`" -O2 -I`"$inc`" `"$lib`""
$results["bwprobe"] = foreach ($i in 1..3) { & "$out\bwprobe.exe" }
if (Get-Command rustc -ErrorAction SilentlyContinue) {
	rustc -O "$PSScriptRoot\ffi_probe.rs" -L (Split-Path $lib) -o "$out\ffi_probe.exe"
	$results["ffi_probe"] = foreach ($i in 1..3) { & "$out\ffi_probe.exe" }
}
$txt = foreach ($k in $results.Keys | Sort-Object) { "== $k"; $results[$k] }
$txt | Set-Content -Encoding utf8 (Join-Path $out "phase0_results.txt")
$txt
