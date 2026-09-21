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


def test_main_check_summary_keeps_raw_records_only_in_appendix(packet):
    from qualification_report import check_summary, details
    checks = [{
        'check': 'Controller pytest', 'outcome': 'passed', 'result': '784 passed; 18 warnings',
        'exit_code': 0, 'peak_rss_kib': 573512, 'wall_seconds': 142.96,
        'sha256': 'b' * 64, 'command': ['uv', 'run', 'pytest'],
        'log': '/external/check.log', 'session': 'session-identity',
        'environment_overrides': {'OMP_NUM_THREADS': '1'}, 'future_raw_field': 'raw-detail',
    }]
    original = deepcopy(checks)
    summary = check_summary(checks)
    appendix = details('Complete check records', checks)
    for value in ('Controller pytest', 'passed', '784 passed; 18 warnings', '573512', '142.96'):
        assert value in summary
    for key in ('sha256', 'command', 'log', 'session', 'environment_overrides', 'future_raw_field'):
        assert key not in summary
        assert key in appendix
    assert 'b' * 64 not in summary and 'b' * 64 in appendix
    assert 'raw-detail' not in summary and 'raw-detail' in appendix
    assert checks == original


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
