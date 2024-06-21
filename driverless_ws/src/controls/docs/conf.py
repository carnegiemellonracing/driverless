# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'driverless_controls'
copyright = '2024, Anthony Yip'
author = 'Anthony Yip'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

master_doc = 'index'

# -- Epilog -----------------------------------------------------------------

rst_epilog = """
.. |Node| replace:: `Node <https://docs.ros.org/en/humble/Concepts/Basic/About-Nodes.html>`__
.. |Path Planning| replace:: Path Planning
.. |twist| replace:: :doc:`twist </source/explainers/terminology>`
.. |Actuators| replace:: Actuators
"""

# -- Breathe configuration ---------------------------------------------------
import os, sys
sys.path.append(os.fspath('../venv/lib/python3.10/site-packages'))
extensions = ['breathe']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

breathe_projects = {"driverless_controls": "doxyxml/"}
breathe_default_project = "driverless_controls"
breathe_default_members = ('members', 'protected-members', 'private-members', 'undoc-members')

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static', 'pdfs']
