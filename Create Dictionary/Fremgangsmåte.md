0. Installer wireshark (med tshark), legg tshark i PATH. Installer Sysmon og kjør ```.\sysmon64.exe -i ja4_config.xml```

1. Start tshark:\
```tshark -i "Wi-Fi" -w network_data.pcapng```
2. Start Sysmon logging:\
```.\getSysmonLog.ps1```
3. Stopp tshark og Sysmon loggin (Ctrl+C)\
2. Konverter network data til json med JA4 (Må kjøres utenfor venv for å få tilgang til tshark path):\
```python .\FoxIO-python\ja4.py .\network_data.pcapng -Jv -f .\network_data.json --ja4 --ja4s```
5. Korreler Sysmon og JA4 data:\
```python .\correlateSysmonNetwork.py```