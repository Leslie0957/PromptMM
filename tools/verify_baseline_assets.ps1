[CmdletBinding()]
param(
    [string]$Manifest = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

if (-not $Manifest) {
    $Manifest = Join-Path $repoRoot 'docs\baselines\ASSET_MANIFEST_2026-07-29.csv'
}

$manifestPath = (Resolve-Path $Manifest).Path
$rows = Import-Csv -LiteralPath $manifestPath
$failures = [System.Collections.Generic.List[string]]::new()
$verified = 0

foreach ($row in $rows) {
    $relativePath = $row.relative_path.Replace('/', [IO.Path]::DirectorySeparatorChar)
    $path = Join-Path $repoRoot $relativePath

    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $failures.Add("MISSING: $($row.relative_path)")
        continue
    }

    $item = Get-Item -LiteralPath $path
    if ($item.Length -ne [int64]$row.bytes) {
        $failures.Add("SIZE: $($row.relative_path) expected=$($row.bytes) actual=$($item.Length)")
        continue
    }

    $actualHash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    if ($actualHash -ne $row.sha256) {
        $failures.Add("SHA256: $($row.relative_path) expected=$($row.sha256) actual=$actualHash")
        continue
    }

    $verified++
}

Write-Host "Verified $verified of $($rows.Count) fingerprinted assets."

if ($failures.Count -gt 0) {
    $failures | ForEach-Object { Write-Error $_ }
    exit 1
}

Write-Host 'Asset verification passed.'
