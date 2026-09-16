# Troubleshooting

## Documentation builds

- **`sphinx-build` is unavailable:** run `uv sync --locked --no-default-groups --group docs`
  from `src/python`, and include `--group docs` in the build command.
- **`starfinder` cannot be imported:** build from `src/python` with `uv run` so the
  checkout and its runtime dependencies are installed. Autodoc uses real imports.
- **A warning makes the build fail:** read the file and line in the build output.
  Fix the reference or markup, then rebuild into a fresh external output directory.
- **Preview port is in use:** choose a different port in the `http.server` command
  and open the matching URL. Stop the server with Ctrl-C when finished.

See [contributing](contributing.md) for the exact commands.

**Coverage planned:** pipeline troubleshooting and diagnostic recipes. This page
currently covers only local documentation setup.
