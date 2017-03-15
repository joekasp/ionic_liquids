# Overview
- This code is to build a regression model, taking the chemical/structure information of the binary ionic liquid and predicting the electric conductivity.

## Input 
- There should be two input data, one is the feature data, one is the target value data.

- The input feature data set will have the following columns:

`SMILES_A | SMILES_B | Mol_ratio_A | Tempurature |Pressure `

- The rdkit functions will convert the SMILES string of A and B molecules to the corresponding molecular descriptor (black boxed in this code) and passing the resulting dataframe to the regression model.

- The input target value data has the same length as the input feature data, the element in the vector are the experimental electrical conductivity.

## Output

- The output value is the prediction of the electrical conductivity.

