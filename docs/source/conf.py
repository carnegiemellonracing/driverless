# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CMR Driverless Path Planning'
copyright = '2025, Carnegie Mellon Racing'
author = 'Carnegie Mellon Racing'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_rtd_theme',
    'breathe',
    'sphinx.ext.autosectionlabel'
]

templates_path = ['_templates']
exclude_patterns = []

breathe_projects = {"iSAM2": "../doxygen_output/xml"}
breathe_default_project = "iSAM2"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
