## How to Generate Docs

- make sure you have the following dependencies installed
  
  - `doxygen` via apt-get
  - `sphinx` and `breathe` via python pip: `pip install Sphinx breath` 
  - sphinx read the docs theme via python pip: `pip install sphinx_rtd_theme`

- `cd` into `point_to_pixel/docs` dir
- run `doxygen Doxyfile`
- run `sphinx-build -b html . _build/`

