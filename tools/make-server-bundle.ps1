# Packs the dedicated server's source into mazebeasts-server-<version>-src.tar.gz, for building
# on Linux (for example on an AWS EC2 instance): everything the Makefile needs in one folder,
# with Unix line endings (make chokes on Windows ones).
#
#   powershell -ExecutionPolicy Bypass -File tools\make-server-bundle.ps1 [-Version 0.4] [-OutDir folder]
param(
    [string]$Version = '0.4',
    [string]$OutDir = (Join-Path $PSScriptRoot '..\x64\Release')
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$name = "mazebeasts-server-$Version"
$stage = Join-Path ([IO.Path]::GetTempPath()) ("mazebeasts-bundle-" + [guid]::NewGuid().ToString('N'))
$dir = Join-Path $stage $name
New-Item -ItemType Directory -Force (Join-Path $dir 'enet\include\enet') | Out-Null

$utf8 = New-Object System.Text.UTF8Encoding $false
function Copy-Unix([string]$src, [string]$dst) {
    $text = [IO.File]::ReadAllText($src) -replace "`r`n", "`n"
    [IO.File]::WriteAllText($dst, $text, $utf8)
}

Copy-Unix "$root\MazeBeastsServer\server.cpp" "$dir\server.cpp"
Copy-Unix "$root\MazeBeastsServer\Makefile"   "$dir\Makefile"
Copy-Unix "$root\MazeBeastsServer\README.md"  "$dir\README.md"
Copy-Unix "$root\LICENSE"                     "$dir\LICENSE"
foreach ($f in 'world.h', 'world.cpp', 'net.h', 'net.cpp', 'protocol.h') {
    Copy-Unix "$root\MazeBeasts\$f" "$dir\$f"
}
foreach ($f in 'callbacks.c', 'compress.c', 'host.c', 'list.c', 'packet.c', 'peer.c', 'protocol.c', 'unix.c', 'LICENSE', 'VERSION.txt') {
    Copy-Unix "$root\MazeBeasts\enet\$f" "$dir\enet\$f"
}
Get-ChildItem "$root\MazeBeasts\enet\include\enet\*.h" | ForEach-Object {
    Copy-Unix $_.FullName "$dir\enet\include\enet\$($_.Name)"
}

New-Item -ItemType Directory -Force $OutDir | Out-Null
$out = Join-Path (Resolve-Path $OutDir).Path "$name-src.tar.gz"
tar -czf $out -C $stage $name   # Windows 10 and 11 include bsdtar
if ($LASTEXITCODE -ne 0) { throw "tar failed" }
Remove-Item -LiteralPath $stage -Recurse -Force
Write-Output $out
