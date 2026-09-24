# Bundle every runtime asset into a single assets.dat, which the game reads at startup.
#
# Usage (from anywhere):  powershell -ExecutionPolicy Bypass -File tools\packassets.ps1
#
# Layout:  "MBPACK01"  magic (8 bytes)
#          uint32      entry count
#          per entry:  uint16 nameLen, name bytes (UTF-8), uint64 offset, uint64 size
#          then the raw file blobs, in entry order.
param(
    [string]$ProjectDir = (Join-Path $PSScriptRoot '..\MazeBeasts'),
    [string]$OutFile    = ''
)
$ProjectDir = (Resolve-Path $ProjectDir).Path
if (-not $OutFile) { $OutFile = Join-Path $ProjectDir 'assets.dat' }

$names = @('LiberationSans-Regular.ttf','wall_texture.png','boundary_texture.png','monster_texture.png',
           'monster2_texture.png','medpack.png','cavewalltexture.jpg','cavefloorgrok.jpg',
           'monster_sound.flac','boss_sound.flac')

$missing = $names | Where-Object { -not (Test-Path (Join-Path $ProjectDir $_)) }
if ($missing) { throw "missing asset(s): $($missing -join ', ')" }

$blobs = @{}
foreach ($n in $names) { $blobs[$n] = [System.IO.File]::ReadAllBytes((Join-Path $ProjectDir $n)) }

# The header has to be sized before blob offsets can be written.
$headerSize = 8 + 4
foreach ($n in $names) { $headerSize += 2 + [System.Text.Encoding]::UTF8.GetByteCount($n) + 8 + 8 }

$fs = [System.IO.File]::Create($OutFile)
$bw = New-Object System.IO.BinaryWriter($fs)
$bw.Write([System.Text.Encoding]::ASCII.GetBytes('MBPACK01'))
$bw.Write([uint32]$names.Count)

$offset = [uint64]$headerSize
foreach ($n in $names) {
    $nb = [System.Text.Encoding]::UTF8.GetBytes($n)
    $bw.Write([uint16]$nb.Length)
    $bw.Write($nb)
    $bw.Write([uint64]$offset)
    $bw.Write([uint64]$blobs[$n].Length)
    $offset += [uint64]$blobs[$n].Length
}
foreach ($n in $names) { $bw.Write($blobs[$n]) }
$bw.Flush(); $bw.Close(); $fs.Close()

"packed $($names.Count) files -> $OutFile ($('{0:N0}' -f (Get-Item $OutFile).Length) bytes)"
