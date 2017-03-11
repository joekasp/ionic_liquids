import pandas as pd
import numpy as np

def data_cleaning(dataframe):

	#Data Cleaning
	data = pd.read_excel("inputdata.xlsx")
	data['EC_value'], data['EC_error'] = zip(*data['ELE_COD'].map(lambda x: x.split('±')))

	return data
