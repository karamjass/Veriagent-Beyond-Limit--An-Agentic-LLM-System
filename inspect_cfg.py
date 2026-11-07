import json

for cfg in (9, 12):
    rows = [json.loads(l) for l in open(f'results/raw/gsm8k_cfg{cfg}_N40_seed42.jsonl')]
    print(f'===== cfg{cfg} =====')
    print('keys:', list(rows[0].keys()))
    for i in range(3):
        r = rows[i]
        for k, v in r.items():
            if isinstance(v, str):
                print(f'--- row {i} | field "{k}" | length {len(v)} | last 200 chars:')
                print(v.strip()[-200:])
        print()