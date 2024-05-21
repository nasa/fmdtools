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
import os
import sys
sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'fmdtools'
copyright = '2024, United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved'
author = 'fmdtools developers'

# The full version, including alpha/beta/rc tags
release = '2.0-rc-4'
version = release


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.intersphinx',
              'sphinx.ext.autosummary',
              "nbsphinx", "myst_parser", "sphinx.ext.githubpages"]

# "gaphor.extensions.sphinx"
# gaphor_models = "/docs/figures/uml/module-reference-diagrams.gaphor"
myst_enable_extensions = ["html_image", "html_admonition"]


exclude_patterns = ['_build', '**.ipynb_checkpoints', 'rad_models*']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {"logo_only": True,
                      "collapse_navigation": False,
                      'display_version': True,
                      'github_url': 'https://github.com/nasa/fmdtools'}

html_favicon = 'docs/figures/logo/fmdtools_ico.ico'

html_logo = 'docs/figures/logo/logo-main.svg'

html_context = {"display_github": True,  # Integrate GitHub
                "github_user": "nasa",  # Username
                "github_repo": "fmdtools",  # Repo name
                "github_version": "main",  # Version
                "conf_py_path": "/",  # Path in the checkout to the docs root
                }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']



# -- Latex Option ---------------

latex_engine = 'xelatex'
