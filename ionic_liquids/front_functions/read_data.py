import pandas as pd
import numpy as np

def read_data(filename)
    """
    Reads data in from given file to Pandas DataFrame

    Inputs
    -------
    filename : string of path to file

    Returns
    ------ 
    df : Pandas DataFrame

    """
    cols = filename.split('.')
    name = cols[0]
    filetype = cols[1]
    if (filetype == 'csv'):
        df = pd.read_csv(filename)
    elif (filetype in ['xls','xlsx']):
        df = pd.read_excel(filename)
    else:
        raise ValueError('Filetype not supported')

    #clean the data if necessary
	data['EC_value'], data['EC_error'] = zip(*data['ELE_COD'].map(lambda x: x.split('±')))

    return df

