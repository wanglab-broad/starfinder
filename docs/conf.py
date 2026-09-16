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
    "sphinx_design",
]
# Parse only project-owned MATLAB code; no MATLAB process or license is needed.
matlab_src_dir = str(Path(__file__).resolve().parents[1] / "src" / "matlab")
matlab_auto_link = None
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
templates_path = ["_templates"]
myst_heading_anchors = 3
# Colon fences let sphinx-design cards carry icon roles in their titles.
myst_enable_extensions = ["colon_fence"]
exclude_patterns = ["README.md", "_build", "Thumbs.db", ".DS_Store"]
autosummary_generate = True
# Types are already documented by the NumPy-style docstrings. Render those as
# text without duplicating annotation cross-references to external inventories.
autodoc_typehints = "none"
# No network inventories or mocked package imports are needed for this build.
napoleon_use_param = False
napoleon_use_rtype = False

html_theme = "pydata_sphinx_theme"
html_title = "STARfinder"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"
html_static_path = ["_static"]
html_css_files = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800"
    "&family=JetBrains+Mono:wght@400;500;600&display=swap",
    "custom.css",
]
html_show_sourcelink = True
# The landing page is a full-width hero and card grid, and top-level guides
# without child pages have no section navigation to show. Pages with children
# keep the section navigation and page table of contents.
html_sidebars = {
    "index": [],
    "getting-started": [],
    "recipes": [],
    "conventions": [],
    "troubleshooting": [],
    "contributing": [],
}
html_theme_options = {
    "logo": {"text": "STARfinder"},
    "navigation_depth": 2,
    "show_toc_level": 2,
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
    "back_to_top_button": True,
    "pygments_light_style": "github-light",
    "pygments_dark_style": "github-dark",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/wanglab-broad/starfinder",
            "icon": "fa-brands fa-github",
        }
    ],
    "secondary_sidebar_items": {"**": ["page-toc", "sourcelink"], "index": []},
    "footer_start": ["copyright"],
    "footer_end": [],
}
copyright = "STARfinder contributors"

# Preview URLs describe a local server, not a public documentation resource.
linkcheck_ignore = [r"http://127\.0\.0\.1(?::\d+)?/"]
linkcheck_timeout = 15
linkcheck_retries = 1
linkcheck_workers = 5


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
    app.connect("html-page-context", _landing_page_class)


def _landing_page_class(app, pagename, templatename, context, doctree):
    """Scope the landing-page styles in ``_static/custom.css`` to the index."""
    if pagename == "index" and "body" in context:
        context["body"] = f'<div class="sf-landing">{context["body"]}</div>'


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
