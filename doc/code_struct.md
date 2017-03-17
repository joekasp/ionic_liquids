# Code Structure
```
|---ionic_liquid (master)
    |---ionic_liquid
        |---__init__.py
        |---version.py
        |---main.py
        |---util.py
        |---datasets
            |---compoundSMILES.xlsx
            |---compounddata.xlsx
        |---examples
            |---Example_Workflow.ipynb
        |---method
            |---__init__.py
            |---method.py
        |---visualization
            |---__init__.py
            |---core.py
            |---plot.py
        |---test
            |---test_utils.py
            |---test_method.py
            |---test_utils.py
        |Interface.ipynb
    |---doc
        |---overview.md
        |---functional_spec.md
        |---code_struct.md
        |---tutorial.md
        |---runcell.png
        |---model_train.png
        |---model_read.png
    |---README.md
    |---LICENSE
    |---setup.py
```

## Directory Summary
- `datasets` contains the downloaded ionic liquids data

- `methods` contains the regression model used in this work

- `visualization` contains the plot function 

- `test` is the folder for unit test

- `Interface.ipynb` is a portable entrance of the interface widgets.

- `doc` contains documents

- `LICENSE` MIT license


