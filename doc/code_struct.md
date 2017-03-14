# Code Structure
```
-ionic_liquid 
    -ionic_liquid
        |__init__.py
        |version.py
        |util.py
        -method
            |method.py
            |__init__.py
        +datasets
        +examples
        -visualization
            |core.py
            |plot.py
            |__init__.py
        |Interface.ipynb
    -doc
        |functional_spec.md
        |code_struct.md (current)
    |README.md
    |LICENSE
    |setup.py
```

## Code Content

### `ionic_liquid/util.py`
Containing the data cleaning function, regression model selection function.

### `ionic_liquid/method/method.py`
Containing the regression model objective.

### `ionic_liquid/visualization/core.py`
Containing the regression model choose function.

### `ionic_liquid/visualization/plot.py`
Containing the parity plot function and error plot function.

### `ionic_liquid/Interface.ipynb`
Containing the GUI of the interaction widget.

