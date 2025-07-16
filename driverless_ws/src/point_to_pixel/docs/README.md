## How to Generate Docs

<<<<<<< HEAD
- make sure you have the following dependencies installed
  
  - `doxygen` via apt-get
  - `sphinx` and `breathe` via python pip: `pip install Sphinx breathe` 
  - sphinx read the docs theme via python pip: `pip install sphinx_rtd_theme`

=======
>>>>>>> c0d67c9b (added readme with commands to build doxygen docs and sphinx docs)
- make sure you have the following dependencies installed
  
  - `doxygen` via apt-get
  - `sphinx` and `breathe` via python pip: `pip install Sphinx breath` 
  - sphinx read the docs theme via python pip: `pip install sphinx_rtd_theme`

- `cd` into `point_to_pixel/docs` dir
- run `doxygen Doxyfile`
- run `sphinx-build -b html . _build/`

<<<<<<< HEAD
## For Editing

- See [reStructuredText Markup Spec](https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#quick-syntax-overview)
- See [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#lists-and-quote-like-blocks)
- See [breathe docs for interfacing with all things doxy + website linking and structure](https://breathe.readthedocs.io/en/latest/class.html#members-example)
=======
>>>>>>> c0d67c9b (added readme with commands to build doxygen docs and sphinx docs)
