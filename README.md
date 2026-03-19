## Active Learning Example: using the digiLab Uncertainty Engine SDK

`mrf_example.ipynb` contains a straight-forward example of automated active learning for a simple 1 input -> 1 output; using candidate nuclear fusion material behaviour as a theme. 

`mrf_example.ipynb` uses helper functions defined in `utils.py`.

The Uncertainty Engine follows a graph-based paradigm, allowing flexible construction of complex machine learning workflows. To make this more accessible, the graph logic is wrapped into reusable functions in `utils.py`, presenting a more familiar functional interface for Python users.

These functions are used throughout the notebook, but you can explore `utils.py` directly to understand the underlying graph-based implementation.

### Set-up:
- Clone this repo
- Set up a virtual environment (or similar) - refer to https://realpython.com/python-virtual-environments-a-primer/ if unsure)
- Install all project dependencies, by entering the following into your terminal: `pip install -r requirements.txt`
- Head to `mrf_example.ipynb` and enjoy!

