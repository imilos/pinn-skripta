# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Физички засноване неуронске мреже'
copyright = '2022-2023, Милош Ивановић'
author = 'Милош Ивановић'

# The short X.Y version
version = ''
# The full version, including alpha/beta/rc tags
release = 'прво'


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
#extensions = [
#]
extensions = ['sphinx.ext.coverage', 'sphinx.ext.doctest', 'sphinx.ext.githubpages', 
              'sphinxcontrib.bibtex']

bibtex_bibfiles = ['refs.bib']

doctest_global_setup = '''
try:
    import _tkinter
except ImportError:
    _tkinter = None
'''

# Enable warnings for all bad cross references. These are turned into errors
# with the -W flag in the Makefile.
nitpicky = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'
# latex_doc = 'lindex'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'sr'

today = '2022.'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

#[restructuredtext parser]
#syntax_highlight = short

# -- Options for HTML output -------------------------------------------------

def setup(app):
    app.add_js_file('copybutton.js')

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'agogo'
html_theme = 'alabaster'
html_theme = 'classic'
html_theme = 'nature'
html_theme = 'pyramid'
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'PINN'

# -- Options for LaTeX output ------------------------------------------------

CCBYSAlogo = './LaTeXcustom/by-sa.eps'

latex_additional_files = [CCBYSAlogo]
latex_engine = 'xelatex'
latex_elements = {
    'fontpkg': r'''
\setmainfont{DejaVu Serif}
\setsansfont{DejaVu Sans}
\setmonofont{DejaVu Sans Mono}
''',
    'preamble': r'''
\usepackage[titles]{tocloft}
\cftsetpnumwidth {1.25cm}\cftsetrmarg{1.5cm}
\setlength{\cftchapnumwidth}{0.75cm}
\setlength{\cftsecindent}{\cftchapnumwidth}
\setlength{\cftsecnumwidth}{1.25cm}
''',
    'fncychap': r'\usepackage[Bjornstrup]{fncychap}',
    'printindex': r'\footnotesize\raggedright\printindex',
    'papersize' : 'b5paper'
}
latex_show_urls = 'footnote'

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#
# latex_logo = None
latex_logo = './LaTeXcustom/bookwheel.jpg'

# If true, show page references after internal links.
#
latex_show_pagerefs = True

# If true, show URL addresses after external links.
#
latex_show_urls = 'footnote' # = 'inline' # = 'no'

# We use False otherwise the module index gets generated twice.
latex_use_modindex = True

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [(master_doc, 'pinn-skripta.tex', 'Физички засноване неуронске мреже', 'Милош Ивановић', 'manual', True)]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, 'PINN', 'Физички засноване неуронске мреже', [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'PINN', 'Физички засноване неуронске мреже', author,
     'PINN', 'One line description of project.', 'Miscellaneous'),
]

# Enable numref
numfig = True
