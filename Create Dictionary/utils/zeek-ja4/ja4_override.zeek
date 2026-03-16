# SecurityOnion ONLY
# /opt/so/saltstack/local/salt/zeek/policy/custom/ja4_options.zeek
# Override JA4 default configurations


redef FINGERPRINT::JA4_raw = T;

redef FINGERPRINT::JA4S_enabled = T;
redef FINGERPRINT::JA4S_raw = T;

redef FINGERPRINT::JA4H_enabled = T;
redef FINGERPRINT::JA4H_raw = T;

redef FINGERPRINT::JA4T_enabled = T;
redef FINGERPRINT::JA4TS_enabled = T;

# After file creation:
# Security Onion Console webui -> Administration -> Configuration
# Enable Advanced mode
# Navigate to zeek -> config -> local -> load
# Add a new entry: custom/ja4_options.zeek
# Save
# sudo salt-call state.apply zeek
# sudo so-zeek-restart