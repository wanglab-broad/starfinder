#!/usr/bin/env bash
# A stub writes only declared placeholder outputs, a log, and a completion record.
set -euo pipefail
python - "$@" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, task_id, token, attempt = sys.argv[1:]
manifest = json.loads(Path(manifest_path).read_text())
task = next(t for t in manifest['tasks'] if t['id'] == task_id)
root = Path(manifest['roots']['output'])
outputs = []
for name in [*task['outputs'], task['log']]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if name.endswith('/'):
        path.mkdir(exist_ok=True)
        outputs.append({'path': name, 'files': []})
    else:
        path.write_text('Nextflow stub: ' + task_id + '\n')
        outputs.append({'path': name, 'bytes': path.stat().st_size,
                        'sha256': 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()})
record = {'status': 'succeeded', 'workflow': {'fingerprint': task['fingerprint'], 'token': token},
          'outputs': outputs, 'stub': True, 'attempt': int(attempt)}
path = root / task['record']
path.parent.mkdir(parents=True, exist_ok=True)
text = json.dumps(record) + '\n'
path.write_text(text)
Path(task_id.replace('/', '.') + '.json').write_text(text)
PY
