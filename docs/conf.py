project = "raycachetools"
copyright = "2026"
author = "Leonard Schulz"
release = "0.1.0"

extensions = ["sphinx.ext.intersphinx"]
exclude_patterns = ["_build"]
master_doc = "index"

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "cachetools": ("https://cachetools.readthedocs.io/en/stable/", None),
    "ray": ("https://docs.ray.io/en/latest/", None),
}
