import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def test_model_types():
    """
    Test choosing the model
    """
    models = ['MLP Regressor', 'MLP Classifier', 'LASSO', 'SVR']
    assert len(models)==4, "Too more models than expect"
    return models


def test_read_SMILES():
    """
    Test Reading compound names and SMILES from file

    Inputs
    -----
    None


    Returns
    ------
    A_list : names of 'A' compounds
    A_smiles : SMILES of corresponding 'A' compounds
    B_list : names of 'B' compounds
    B_smiles : SMILES of corresponding 'B' compounds
    
    Check:
    1. Make sure the xlsx is read to a dataframe, otherwise 
    the itertuples() will cast ans error
    """

    df = pd.read_excel('../datasets/compoundSMILES.xlsx')
    A_list = []
    A_smiles = []
    B_list = []
    B_smiles = []
    assert isinstance(df,pd.DataFrame), "You should read the xlsx to a pandas dataframe"
    # remove duplicates
    for row in df.itertuples():
        if row[1] not in A_list:
            A_list.append(row[1])
            A_smiles.append(row[2])
        if row[3] not in B_list:
            B_list.append(row[3])
            B_smiles.append(row[4])

    return A_list, A_smiles, B_list, B_smiles
