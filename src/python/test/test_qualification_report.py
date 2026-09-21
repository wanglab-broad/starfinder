"""Delivery-state regressions without regenerating numerical evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest


@pytest.fixture
def packet(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / 'docs/examples'))
    from qualification_report import publication_context
    context = {
        'code': {'commit': 'a' * 40, 'dirty': True},
        'acceptance': [
            {'deliverable': 'Review packet', 'status': 'Old draft'},
            {'deliverable': 'Full gates / local commit', 'status': 'Pending; uncommitted'},
            {'deliverable': 'Independent checks', 'status': '63 presets passed'},
        ],
        'checks': [
            {'check': 'Controller gates', 'outcome': 'Full checks pending'},
            {'check': 'Oracle', 'outcome': 'passed'},
        ],
    }
    return publication_context, context


def final_context(context):
    context = deepcopy(context)
    context['phase'] = 'final'
    context['code']['dirty'] = False
    commands = [['uv', 'run', 'pytest', 'test/', '-v'],
                ['uv', 'run', 'sphinx-build', '-n', '-W'],
                ['uv', 'run', 'python', '../../docs/check_reference.py']]
    context['controller_checks'] = [
        {'command': command, 'log': f'check-{i}.log', 'sha256': str(i) * 64, 'exit_code': 0}
        for i, command in enumerate(commands)
    ]
    return context


def test_draft_does_not_claim_final_validation(packet):
    prepare, context = packet
    original = deepcopy(context)
    embedded, opening, appendix = prepare(context)
    assert 'uncommitted' in opening and 'gates are pending' in opening
    assert 'subsequent steps' in appendix
    assert embedded['phase'] == 'implementation'
    assert context == original


def test_final_replaces_stale_draft_tables_and_embedded_state(packet):
    prepare, context = packet
    context = final_context(context)
    original = deepcopy(context)
    embedded, opening, appendix = prepare(context)
    assert 'all three controller gates passed' in opening
    assert 'W-174 human approval remains pending' in appendix
    payload = json.dumps(embedded) + opening + appendix
    for stale in ('uncommitted', 'Full checks pending', 'Old draft', 'subsequent steps'):
        assert stale not in payload
    assert 'a' * 40 in payload
    assert embedded['controller_checks'] == context['controller_checks']
    assert '63 presets passed' in payload
    assert context == original


@pytest.mark.parametrize('invalid', ['dirty', 'short_commit', 'missing', 'failed', 'missing_log',
                                    'missing_hash', 'duplicate_gate', 'non_strict', 'phase'])
def test_final_rejects_incomplete_or_inconsistent_evidence(packet, invalid):
    prepare, context = packet
    context = final_context(context)
    if invalid == 'dirty':
        context['code']['dirty'] = True
    elif invalid == 'short_commit':
        context['code']['commit'] = 'a' * 12
    elif invalid == 'missing':
        context['controller_checks'].pop()
    elif invalid == 'failed':
        context['controller_checks'][0]['exit_code'] = 1
    elif invalid == 'missing_log':
        context['controller_checks'][0].pop('log')
    elif invalid == 'missing_hash':
        context['controller_checks'][0].pop('sha256')
    elif invalid == 'duplicate_gate':
        context['controller_checks'][2] = context['controller_checks'][0]
    elif invalid == 'non_strict':
        context['controller_checks'][1]['command'].remove('-W')
    else:
        context['phase'] = 'approved'
    with pytest.raises(ValueError):
        prepare(context)
