import json

d = json.load(open('../Dictionary/ja4+_db.json'))

apps = [str(x.get('application', '')).lower() for x in d]
uas = [str(x.get('user_agent_string', '')).lower() for x in d]
notes = [str(x.get('notes', '')).lower() for x in d]

print(f"Total apps with 'bot': {sum(1 for a in apps if 'bot' in a)}")
print(f"Total UA with 'bot': {sum(1 for u in uas if 'bot' in u)}")
print(f"Total notes with 'bot': {sum(1 for n in notes if 'bot' in n)}")

print(f"\nTotal apps with 'googlebot': {sum(1 for a in apps if 'googlebot' in a)}")
print(f"Total UA with 'googlebot': {sum(1 for u in uas if 'googlebot' in u)}")

print(f"\nTotal apps with 'bingbot': {sum(1 for a in apps if 'bingbot' in a)}")
print(f"Total UA with 'bingbot': {sum(1 for u in uas if 'bingbot' in u)}")

print(f"\nTotal apps with 'linkedinbot': {sum(1 for a in apps if 'linkedinbot' in a)}")
print(f"Total UA with 'linkedinbot': {sum(1 for u in uas if 'linkedinbot' in u)}")
