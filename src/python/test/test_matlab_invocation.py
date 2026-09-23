"""Host selection and error propagation for the shared Snakemake launcher."""
import ast
from pathlib import Path
from unittest.mock import patch
import os
import subprocess

import pytest


def launcher():
    source = Path(__file__).resolve().parents[3] / 'workflow/rules/common.smk'
    tree = ast.parse(source.read_text().split('rule rsf_preparation:')[0])
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_matlab_scripts')
    namespace = {'config': {'starfinder_path': "/repo's path"}, 'Path': Path, 'os': os}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['run_matlab_scripts']


def test_local_launcher_quotes_and_propagates(monkeypatch):
    monkeypatch.setenv('STARFINDER_MATLAB_EXECUTABLE', '/local/matlab')
    with patch('shutil.which', return_value='/local/matlab'), patch('subprocess.run') as run:
        launcher()("'literal $HOME `value`'", 'rsf_single_fov')
        argv = run.call_args.args[0]
        assert argv[:3] == ['/local/matlab', '-singleCompThread', '-batch']
        assert "addpath('/repo''s path/workflow/scripts')" in argv[3]
        assert "'literal $HOME `value`'" in argv[3]
        assert run.call_args.kwargs == {'check': True}
        run.side_effect = subprocess.CalledProcessError(1, argv)
        with pytest.raises(subprocess.CalledProcessError): launcher()('', 'rsf_single_fov')


def test_explicit_missing_never_falls_back(monkeypatch):
    monkeypatch.setenv('STARFINDER_MATLAB_EXECUTABLE', '/missing/matlab')
    with patch('shutil.which', return_value=None), patch('subprocess.run') as run:
        with pytest.raises(FileNotFoundError, match='Selected'): launcher()('', 'rsf_single_fov')
        run.assert_not_called()


def test_broad_fallback_and_missing(monkeypatch):
    monkeypatch.delenv('STARFINDER_MATLAB_EXECUTABLE', raising=False)
    launch = launcher()
    with patch('shutil.which', return_value=None), patch.object(Path, 'is_file', return_value=True), patch('subprocess.run') as run:
        launch('', 'rsf_single_fov')
        assert run.call_args.args[0][:2] == ['/bin/bash', '-c']
        assert 'use Matlab && exec matlab -singleCompThread -batch' in run.call_args.args[0][2]
    with patch('shutil.which', return_value=None), patch.object(Path, 'is_file', return_value=False):
        with pytest.raises(FileNotFoundError, match='MATLAB unavailable'): launch('', 'rsf_single_fov')
    with pytest.raises(ValueError): launch('', 'invalid;exit')
