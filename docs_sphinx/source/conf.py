# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import sys
modulepath = os.path.abspath('../..')
sys.path.insert(0, modulepath)
print("* USING conf.py from:", __file__)
print('*'*80,modulepath,'*'*80,sep="\n")
#import sig2dna_core.signomics

# -- Project information -----------------------------------------------------

project = '🤖 GS-HostAgents'
copyright = '2025, Olivier Vitrac (Generative Simulation)'
author = 'Olivier Vitrac (Generative Simulation)'

# The full version, including alpha/beta/rc tags
release = '0.15'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "processEscapes": True,
        "tags": "ams",
    }
}

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # For Google/Numpy style docstrings
    'myst_parser',          # Optional, if you use Markdown in docs
    'sphinxcontrib.mermaid',
]
extensions += ['sphinx.ext.mathjax']

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
    "attrs_inline",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
#html_theme = 'sphinx_rtd_theme'  # Or 'furo', 'alabaster', etc.

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
