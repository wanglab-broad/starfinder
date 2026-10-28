"""Sphinx configuration for the maintained STARfinder documentation."""

from importlib.metadata import version as package_version
import posixpath
import re
from pathlib import Path
import sys

# Project-owned Sphinx extensions (docs/_ext).
sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))

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
    "synthetic_presets",
    "method_tables",
    "report_tables",
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
autodoc_member_order = "alphabetical"
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
# The landing page is a full-width hero and card grid. Every other page sits in
# one of the five sections and keeps the section navigation. The section and
# subsection index pages are maps of cards, so they drop the page table of
# contents and use its width.
SECTION_PAGES = [
    "get-started/index", "workflow/index", "api/index", "benchmark/index", "wiki",
    "wiki/design", "wiki/pipeline-stages", "wiki/dataset", "wiki/synthetic",
    "wiki/real-data", "wiki/development",
]
html_sidebars = {
    "index": [],
}
html_theme_options = {
    "logo": {"text": "STARfinder"},
    # Four sidebar levels reach a stage's pages in the Wiki section: Starfinder
    # Design › Pipeline stages › the stage guide › its algorithms and baseline.
    "navigation_depth": 4,
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
    "secondary_sidebar_items": {
        "**": ["page-toc", "sourcelink"], "index": [], **{page: [] for page in SECTION_PAGES}},
    "footer_start": ["copyright"],
    "footer_end": [],
}
copyright = "STARfinder contributors"

# Preview URLs describe a local server, not a public documentation resource.
linkcheck_ignore = [r"http://127\.0\.0\.1(?::\d+)?/"]
linkcheck_timeout = 15
linkcheck_retries = 1
linkcheck_workers = 5


def setup(app):
    app.connect("doctree-read", _cards_follow_toctree)
    app.connect("html-page-context", _source_backlinks)
    app.connect("html-page-context", _landing_page_class)


def _landing_page_class(app, pagename, templatename, context, doctree):
    """Scope the landing and section-page styles in ``_static/custom.css``."""
    if "body" not in context:
        return
    if pagename == "index":
        context["body"] = f'<div class="sf-landing">{context["body"]}</div>'
    elif pagename in SECTION_PAGES:
        context["body"] = f'<div class="sf-section">{context["body"]}</div>'


def _cards_follow_toctree(app, doctree):
    """Warn when a section page's linked cards and its hidden toctree differ.

    A section page shows its subsections as sphinx-design cards and keeps the
    navigation in a hidden toctree; both must name the same pages in one order.
    """
    from sphinx import addnodes
    from sphinx.util import docname_join, logging

    docname = app.env.docname
    if docname not in SECTION_PAGES:
        return
    cards = [docname_join(docname, node["reftarget"])
             for node in doctree.findall(addnodes.pending_xref)
             if "sd-stretched-link" in node.get("classes", []) and node.get("reftype") == "doc"]
    if not cards:
        return
    entries = [entry for node in doctree.findall(addnodes.toctree) if node.get("hidden")
               for _, entry in node["entries"]]
    if cards != entries:
        logging.getLogger(__name__).warning(
            "section cards %s differ from the hidden toctree %s", cards, entries,
            location=docname)


def _source_backlinks(app, pagename, templatename, context, doctree):
    """Resolve viewcode backlinks when one source module has several aliases.

    Sphinx 8 viewcode keeps one import prefix per source module. Public
    namespaces re-export private implementations, so the generated backlink
    prefix can be wrong. Use the documented object's
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
