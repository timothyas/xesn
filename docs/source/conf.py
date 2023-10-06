# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath("../"))

project = 'esnpy'
copyright = '2023, Timothy A. Smith, Stephen G. Penny, Jason A. Platt, Tse-Chun Chen'
author = 'Timothy A. Smith, Stephen G. Penny, Jason A. Platt, Tse-Chun Chen'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        "sphinx.ext.autodoc",
        "sphinx.ext.autosummary",
        "sphinx.ext.napoleon",
        ]

numpydoc_show_class_members = False

templates_path = ['_templates']
exclude_patterns = []

napoleon_custom_sections = [("Returns", "params_style"),
                            ("Sets Attributes", "params_style"),
                            ]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
