"""Sphinx configuration; the documentation build needs no GPU libraries."""
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
project = "SPINNME"
author = "Robert Jarolim"
copyright = "2026, Robert Jarolim"
release = re.search(r'__version__ = "([^"]+)"', (Path(__file__).resolve().parents[1] / "src/spinnme/__init__.py").read_text()).group(1)
extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.autosectionlabel", "sphinx_copybutton"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
autosectionlabel_prefix_document = True
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "pydata_sphinx_theme"
templates_path = ["_templates"]
html_sidebars = {"**": ["documentation-nav.html"], "index": []}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "assets/SPINNME_logo.png"
html_favicon = "assets/SPINNME_favicon.png"
html_title = "SPINNME · Spherical solar inversions"
html_theme_options = {
    "logo": {"alt_text": "SPINNME · Spherical solar inversions"},
    "show_prev_next": True,
    "navigation_depth": 2,
    "show_toc_level": 2,
    "header_links_before_dropdown": 4,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [{"name": "GitHub", "url": "https://github.com/RobertJaro/spinnme", "icon": "fa-brands fa-github"}],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}
copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True
