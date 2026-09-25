"""Config-driven MATLAB launcher used by the Snakemake rules; MATLAB is never executed."""
import importlib.util
import shlex
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('matlab_launcher', ROOT / 'workflow/rules/matlab_launcher.py')
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)

CONFIG = {'starfinder_path': "/repo's path"}
PARAMS = "'literal $HOME `value`', 'tile_1'"
EXPRESSION = "addpath('/repo''s path/workflow/scripts'); rsf_single_fov('literal $HOME `value`', 'tile_1');"


@pytest.fixture
def local_matlab(monkeypatch):
    monkeypatch.delenv(launcher.EXECUTABLE_VARIABLE, raising=False)
    with patch('shutil.which', return_value='/opt/matlab/bin/matlab') as which, \
            patch('subprocess.run') as run:
        yield which, run


def test_existing_config_defaults_to_path_multithreaded(local_matlab, capsys):
    which, run = local_matlab
    launcher.run_matlab_scripts(CONFIG, PARAMS, 'rsf_single_fov')
    which.assert_called_once_with('matlab')
    argv = ['/opt/matlab/bin/matlab', '-batch', EXPRESSION]
    run.assert_called_once_with(argv, check=True)
    assert capsys.readouterr().out == shlex.join(argv) + '\n'


def test_path_launcher_uses_selected_executable(monkeypatch):
    monkeypatch.setenv(launcher.EXECUTABLE_VARIABLE, '/local/R2024a/bin/matlab')
    with patch('shutil.which', return_value='/local/R2024a/bin/matlab') as which:
        argv = launcher.matlab_command({**CONFIG, 'matlab_launcher': 'path'}, '', 'rsf_single_fov')
    which.assert_called_once_with('/local/R2024a/bin/matlab')
    assert argv[:2] == ['/local/R2024a/bin/matlab', '-batch']


def test_single_thread_is_opt_in(local_matlab):
    config = {**CONFIG, 'matlab_single_thread': True}
    assert launcher.matlab_command(config, PARAMS, 'rsf_single_fov')[1:] == \
        ['-singleCompThread', '-batch', EXPRESSION]
    config['matlab_single_thread'] = False
    assert '-singleCompThread' not in launcher.matlab_command(config, PARAMS, 'rsf_single_fov')
    config['matlab_single_thread'] = 'yes'
    with pytest.raises(ValueError, match='matlab_single_thread'):
        launcher.matlab_command(config, PARAMS, 'rsf_single_fov')


@pytest.mark.parametrize('single_thread', [False, True])
def test_broad_launcher_loads_matlab_in_same_shell(single_thread, capsys):
    config = {**CONFIG, 'matlab_launcher': 'broad', 'matlab_single_thread': single_thread}
    with patch.object(Path, 'is_file', return_value=True), patch('shutil.which') as which, \
            patch('subprocess.run') as run:
        launcher.run_matlab_scripts(config, PARAMS, 'rsf_single_fov')
    which.assert_not_called()
    argv = run.call_args.args[0]
    assert argv[:2] == ['/bin/bash', '-c'] and run.call_args.kwargs == {'check': True}
    flags = ['-singleCompThread'] if single_thread else []
    assert argv[2] == ('source /broad/software/scripts/useuse && use Matlab && exec '
                       + shlex.join(['matlab', *flags, '-batch', EXPRESSION]))
    # The shell parses the MATLAB expression back unchanged.
    assert shlex.split(argv[2].split(' && exec ')[1])[-1] == EXPRESSION
    assert capsys.readouterr().out == shlex.join(argv) + '\n'


def test_matlab_failure_propagates(local_matlab):
    _, run = local_matlab
    run.side_effect = subprocess.CalledProcessError(1, ['matlab'])
    with pytest.raises(subprocess.CalledProcessError):
        launcher.run_matlab_scripts(CONFIG, '', 'rsf_single_fov')


def test_missing_executables_raise_without_running(monkeypatch):
    monkeypatch.delenv(launcher.EXECUTABLE_VARIABLE, raising=False)
    with patch('shutil.which', return_value=None), patch('subprocess.run') as run:
        with pytest.raises(FileNotFoundError, match='MATLAB not found'):
            launcher.run_matlab_scripts(CONFIG, '', 'rsf_single_fov')
        monkeypatch.setenv(launcher.EXECUTABLE_VARIABLE, '/missing/matlab')
        with pytest.raises(FileNotFoundError, match='/missing/matlab'):
            launcher.run_matlab_scripts(CONFIG, '', 'rsf_single_fov')
        with patch.object(Path, 'is_file', return_value=False):
            with pytest.raises(FileNotFoundError, match='useuse'):
                launcher.run_matlab_scripts({**CONFIG, 'matlab_launcher': 'broad'}, '', 'rsf_single_fov')
    run.assert_not_called()


@pytest.mark.parametrize('config, name, message', [
    (CONFIG, 'invalid;exit', 'function name'),
    (CONFIG, "rsf_single_fov('x')", 'function name'),
    (CONFIG, '1rsf', 'function name'),
    (CONFIG, '', 'function name'),
    ({**CONFIG, 'matlab_launcher': 'uger'}, 'rsf_single_fov', 'Unknown matlab_launcher'),
])
def test_invalid_names_and_launchers_raise_without_running(config, name, message):
    with patch('subprocess.run') as run, pytest.raises(ValueError, match=message):
        launcher.run_matlab_scripts(config, '', name)
    run.assert_not_called()


def test_rules_delegate_to_the_launcher():
    common = (ROOT / 'workflow/rules/common.smk').read_text()
    assert 'matlab_launcher.run_matlab_scripts(config, param_string, matlab_script_name)' in common
    assert 'useuse' not in common


def test_schema_declares_launcher_defaults():
    properties = yaml.safe_load((ROOT / 'workflow/schemas/config.schema.yaml').read_text())['properties']
    assert properties['matlab_launcher']['enum'] == list(launcher.LAUNCHERS)
    assert properties['matlab_launcher']['default'] == launcher.DEFAULT_LAUNCHER
    assert properties['matlab_single_thread'] == {**properties['matlab_single_thread'],
                                                  'type': 'boolean', 'default': False}
