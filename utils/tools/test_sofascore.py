import subprocess
import json
import sys

API_KEY = 'c2e4c0828cmsh0e044c60fbedf45p1e0984jsn12eecacdf11d'
HOST = 'sofascore6.p.rapidapi.com'
BASE = 'https://sofascore6.p.rapidapi.com/api/v1'

headers_args = [
    '-H', 'x-rapidapi-key: ' + API_KEY,
    '-H', 'x-rapidapi-host: ' + HOST
]

def test_endpoint(name, url):
    sep = '='*60
    print('\n' + sep)
    print('TEST: ' + name)
    print('URL: ' + url)
    cmd = ['curl', '-s', '-w', '\nHTTP_STATUS:%{http_code}', url] + headers_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    if '\nHTTP_STATUS:' in output:
        body, status = output.rsplit('\nHTTP_STATUS:', 1)
        print('HTTP Status: ' + status.strip())
    else:
        body = output
        print('HTTP Status: unknown')

    try:
        data = json.loads(body)
        if isinstance(data, dict):
            keys = list(data.keys())
        else:
            keys = 'LIST len=' + str(len(data))
        print('Top-level keys: ' + str(keys))

        if isinstance(data, dict):
            for k, v in list(data.items())[:5]:
                if isinstance(v, list) and len(v) > 0:
                    fk = list(v[0].keys()) if isinstance(v[0], dict) else str(v[0])
                    print('  [' + k + '] list len=' + str(len(v)) + ', first item keys: ' + str(fk))
                    if isinstance(v[0], dict):
                        sample = json.dumps(v[0], indent=2)
                        print('    Sample (first 600 chars):\n' + sample[:600])
                elif isinstance(v, dict):
                    print('  [' + k + '] dict keys: ' + str(list(v.keys())[:8]))
                else:
                    print('  [' + k + ']: ' + str(v)[:200])
        elif isinstance(data, list) and len(data) > 0:
            print('  First item: ' + json.dumps(data[0], indent=2)[:500])
        return data
    except Exception as e:
        print('Parse error: ' + str(e))
        print('Raw (first 400): ' + body[:400])
        return None

# ---- TEST 1: Scheduled events today ----
d1 = test_endpoint('Scheduled events today (2026-03-10)',
                   BASE + '/sport/football/scheduled-events/2026-03-10')

# ---- TEST 2: Scheduled events tomorrow ----
d2 = test_endpoint('Scheduled events tomorrow (2026-03-11)',
                   BASE + '/sport/football/scheduled-events/2026-03-11')

# ---- TEST 3: Team last events (Atalanta id=2686) ----
d3 = test_endpoint('Team last events - Atalanta (2686)',
                   BASE + '/team/2686/events/last/0')

# ---- TEST 4: Team next events (Atalanta) ----
d4 = test_endpoint('Team next events - Atalanta (2686)',
                   BASE + '/team/2686/events/next/0')

# ---- Grab a match ID from test 1 or test 7 for stats/odds ----
print('\n' + '='*60)
print('Looking for match IDs from today/tomorrow...')
match_id = None
for d in [d1, d2]:
    if d and isinstance(d, dict) and 'events' in d:
        events = d['events']
        if events and len(events) > 0:
            match_id = events[0].get('id')
            print('Found match_id: ' + str(match_id) + ' from ' + str(events[0].get('homeTeam', {}).get('name', '?')) + ' vs ' + str(events[0].get('awayTeam', {}).get('name', '?')))
            break

# ---- TEST 6: UCL seasons ----
d6 = test_endpoint('UCL unique tournament seasons (id=7)',
                   BASE + '/unique-tournament/7/seasons')

# ---- TEST 7: UCL season last events ----
d7 = test_endpoint('UCL season 76953 last events',
                   BASE + '/unique-tournament/7/season/76953/events/last/0')

# Get a match id from UCL events if not already found
ucl_match_id = None
if d7 and isinstance(d7, dict) and 'events' in d7:
    ev = d7['events']
    if ev and len(ev) > 0:
        ucl_match_id = ev[0].get('id')
        print('\nUCL match_id found: ' + str(ucl_match_id))

# ---- TEST 8: UCL standings ----
d8 = test_endpoint('UCL season 76953 standings',
                   BASE + '/unique-tournament/7/season/76953/standings/total')

# ---- TEST 9: Team season stats ----
d9 = test_endpoint('Atalanta UCL season stats',
                   BASE + '/team/2686/unique-tournament/7/season/76953/statistics/overall')

# ---- TEST 5: Match statistics (use match_id if found) ----
use_id = match_id or ucl_match_id
if use_id:
    d5 = test_endpoint('Match statistics (id=' + str(use_id) + ')',
                       BASE + '/event/' + str(use_id) + '/statistics')
else:
    print('\nNo match ID found for statistics test, skipping test 5')

# ---- TEST 10: Event odds ----
odds_id = ucl_match_id or match_id
if odds_id:
    d10 = test_endpoint('Event odds (id=' + str(odds_id) + ')',
                        BASE + '/event/' + str(odds_id) + '/odds/1/all')
else:
    print('\nNo match ID found for odds test')

print('\n' + '='*60)
print('ALL TESTS COMPLETE')
