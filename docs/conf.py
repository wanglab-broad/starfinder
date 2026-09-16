"""Sphinx configuration for the maintained STARfinder documentation."""

from importlib.metadata import version as package_version
import posixpath
import re
from pathlib import Path

project = "STARfinder"
author = "STARfinder contributors"
release = package_version("starfinder")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.matlab",
]
# Parse only project-owned MATLAB code; no MATLAB process or license is needed.
matlab_src_dir = str(Path(__file__).resolve().parents[1] / "src" / "matlab")
matlab_auto_link = None
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


def _alias_source_module(app, modname, attribute):
    """Keep type-alias source lookup in STARfinder, not builtins/NumPy."""
    if modname == "starfinder.dataset" and attribute in {
        "Shift3D", "ImageArray", "ChannelOrder"
    }:
        return "starfinder.dataset.types"
    if modname == "starfinder.benchmark.synthetic" and attribute == "SpotTuple":
        return modname
    return None


def setup(app):
    app.connect("viewcode-follow-imported", _alias_source_module)
    app.connect("html-page-context", _source_backlinks)


def _source_backlinks(app, pagename, templatename, context, doctree):
    """Resolve viewcode backlinks when one source module has several aliases.

    Sphinx 8 viewcode keeps one import prefix per source module. Point-set
    helpers use direct imports while the high-level API uses re-exports, so
    the generated backlink prefix can be wrong. Use the documented object's
    actual anchor on the already-selected destination page.
    """
    if not pagename.startswith("_modules/") or "body" not in context:
        return
    module = pagename.removeprefix("_modules/").replace("/", ".")
    entry = getattr(app.env, "_viewcode_modules", {}).get(module)
    if not entry:
        return
    prefix = entry[3] + "."

    def resolve(match):
        target, anchor = match[2].split("#", 1)
        docname = posixpath.normpath(posixpath.join(posixpath.dirname(pagename), target))
        docname = docname.removesuffix(app.builder.out_suffix)
        local_name = anchor.removeprefix(prefix)
        for name, obj in app.env.domains["py"].objects.items():
            if obj.docname == docname and name.endswith("." + local_name):
                return match[1] + target + "#" + obj.node_id + match[3]
        return match[0]

    context["body"] = re.sub(
        r'(<a class="viewcode-back" href=")([^"#]+#[^"]+)(")',
        resolve,
        context["body"],
    )
