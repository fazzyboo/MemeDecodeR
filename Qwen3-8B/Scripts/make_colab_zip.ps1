# Build Replication.zip for upload to Google Drive / Colab.
#
# NOTE: this deliberately does NOT use Compress-Archive. Windows PowerShell 5.1 writes
# ZIP entry names with backslash separators ("Dataset\Img\x.jpg"), which violates the ZIP
# spec. Linux unzip (which is what Colab runs) then treats each name as a single flat
# filename containing literal backslashes, so the directory structure is never recreated
# and the notebook cannot find Dataset/Img. Building the archive through .NET and writing
# the entry names ourselves guarantees forward slashes.
#
# Excludes .cache/ (model downloads, ~1 GB, re-downloaded on Colab) and Saved_Models/.

param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot),
    [string]$Out = "D:\MemeDecode\Replication.zip"
)

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

if (Test-Path $Out) { Remove-Item $Out -Force }

$include = @("Dataset", "Scripts")
$includeFiles = @("requirements.txt", "README.md")
$exclude = @("\.cache\\", "\\__pycache__\\", "\.pyc$", "Saved_Models")

$files = @()
foreach ($dir in $include) {
    $p = Join-Path $Root $dir
    if (Test-Path $p) { $files += Get-ChildItem $p -Recurse -File }
}
foreach ($f in $includeFiles) {
    $p = Join-Path $Root $f
    if (Test-Path $p) { $files += Get-Item $p }
}

$files = $files | Where-Object {
    $full = $_.FullName
    $keep = $true
    foreach ($pat in $exclude) { if ($full -match $pat) { $keep = $false } }
    $keep
}

Write-Output ("Adding {0} files..." -f $files.Count)

$zip = [System.IO.Compression.ZipFile]::Open($Out, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    $n = 0
    foreach ($f in $files) {
        # Relative path with FORWARD slashes - this is the whole point of the script.
        $rel = $f.FullName.Substring($Root.Length).TrimStart('\', '/').Replace('\', '/')
        # JPEGs are already compressed; Fastest is much quicker for near-identical size.
        $level = if ($f.Extension -match '^\.(jpg|jpeg|png)$') {
            [System.IO.Compression.CompressionLevel]::Fastest
        } else {
            [System.IO.Compression.CompressionLevel]::Optimal
        }
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $f.FullName, $rel, $level) | Out-Null
        $n++
        if ($n % 500 -eq 0) { Write-Output ("  {0}/{1}" -f $n, $files.Count) }
    }
} finally {
    $zip.Dispose()
}

$z = Get-Item $Out
Write-Output ("Created: {0}  ({1} MB)" -f $z.FullName, [math]::Round($z.Length / 1MB, 1))
