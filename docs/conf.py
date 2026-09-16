"""Sphinx configuration for the maintained STARfinder documentation."""

from importlib.metadata import version as package_version

project = "STARfinder"
author = "STARfinder contributors"
release = package_version("starfinder")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
templates_path = ["_templates"]
myst_heading_anchors = 3
exclude_patterns = ["README.md", "_build", "Thumbs.db", ".DS_Store"]
autosummary_generate = True
# Types are already documented by the NumPy-style docstrings. Render those as
# text without duplicating annotation cross-references to external inventories.
autodoc_typehints = "none"
# No network inventories or mocked package imports are needed for this build.
napoleon_use_param = False
napoleon_use_rtype = False

html_theme = "pydata_sphinx_theme"
html_title = "STARfinder documentation"
html_theme_options = {"navigation_depth": 2}
html_show_sourcelink = True
