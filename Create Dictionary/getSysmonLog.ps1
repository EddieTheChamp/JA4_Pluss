
$Events = Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-Sysmon/Operational'; ID=3} -MaxEvents 500 -ErrorAction SilentlyContinue | ForEach-Object {

    [PSCustomObject]@{
        timestamp = $_.TimeCreated.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffffffZ")
        app       = $_.Properties[4].Value
        srcport   = $_.Properties[11].Value
        dstport   = $_.Properties[16].Value
        dst       = $_.Properties[14].Value
        domain    = $_.Properties[15].Value
    }
}

$Events | Export-Csv -Path "sysmon_data.csv" -NoTypeInformation