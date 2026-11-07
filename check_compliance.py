import json, re

for cfg in (1, 9, 12):
    rows = [json.loads(l) for l in open(f'results/raw/gsm8k_cfg{cfg}_N40_seed42.jsonl')]
    ok = 0
    for r in rows:
        text = max((v for v in r.values() if isinstance(v, str)), key=len)
        tail = text.strip()[-150:]
        if re.search(r'final answer', tail, re.I):
            ok += 1
    print(f'cfg{cfg}: {ok}/{len(rows)} end with a Final Answer line')
    