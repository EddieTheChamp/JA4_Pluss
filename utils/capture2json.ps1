<#
This script captures network traffic for 60 seconds and converts it to JSON using Zeek and zeek2jsonJA4.py

Usage:
	.\capture2json.ps1 -Interface "Wi-Fi" -Application "MyApp"

Parameters:
	-Interface: Network interface to capture from (required).
	-Application: Name used for generated output files (required).
#>

param(
	[Parameter(Mandatory = $true)]
	[string]$Interface,

	[Parameter(Mandatory = $true)]
	[string]$Application
    
)

$tsharkPath = "C:\Program Files\Wireshark\tshark.exe"

if (-not (Test-Path $tsharkPath)) {
	Write-Error "tshark not found at '$tsharkPath'. Install Wireshark/tshark or update the path in this script."
	exit 1
}

$scriptDir = $PSScriptRoot
$folderPath = Join-Path $scriptDir ("{0}_{1}" -f $Application, (Get-Date -Format "yyyyMMdd_HHmmss"))
$captureFile = "${Application}_capture.pcapng"
$jsonOutput = "${Application}.json"
$capturePath = Join-Path $folderPath $captureFile
$jsonPath = Join-Path $folderPath $jsonOutput
$sslLogPath = Join-Path $folderPath "ssl.log"
$connLogPath = Join-Path $folderPath "conn.log"
$zeek2jsonPath = Join-Path $scriptDir "zeek2jsonJA4.py"

New-Item -ItemType Directory -Path $folderPath -Force | Out-Null

if (-not (Test-Path $zeek2jsonPath)) {
	Write-Error "zeek2jsonJA4.py not found at '$zeek2jsonPath'."
	exit 1
}

& $tsharkPath -i $Interface -a duration:60 -w $capturePath
docker run -it --rm -v "${folderPath}:/data/" -w /data/ zeek-ja4 -C -r "/data/$captureFile" local
py $zeek2jsonPath -a $Application --ssl $sslLogPath --conn $connLogPath > $jsonPath

$filesToKeep = @($capturePath, $jsonPath, $sslLogPath, $connLogPath)

Get-ChildItem -Path $folderPath -File -Recurse -Force |
	Where-Object { $_.FullName -notin $filesToKeep } |
	Remove-Item -Force