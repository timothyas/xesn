# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import datetime
sys.path.insert(0, os.path.abspath("../"))

project = 'xesn'
copyright = '2023, xesn developers'
copyright = f"2023-{datetime.datetime.now().year}, xesn developers"
author = 'xesn developers'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        "sphinx.ext.autodoc",
        "sphinx.ext.autosummary",
        "sphinx.ext.napoleon",
        "nbsphinx",
        "sphinxcontrib.bibtex",
        ]

numpydoc_show_class_members = False
napolean_google_docstring = True
napolean_numpy_docstring = False

templates_path = ['_templates']
exclude_patterns = []

napoleon_custom_sections = [("Returns", "params_style"),
                            ("Sets Attributes", "params_style"),
                            ("Required Parameter Sections", "params_style"),
                            ("Assumptions", "notes_style"),
                            ("Example Config YAML File", "example"),
                            ]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

bibtex_bibfiles = ['references.bib']
bibtex_reference_style = "author_year"

html_theme_options = {
    "repository_url": "https://github.com/timothyas/xesn",
    "use_repository_button": True,
}
