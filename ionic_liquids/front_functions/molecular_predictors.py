#import the packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator as Calculator

def molecular_descriptors(DataFrame):
	'''Read the input csv file,post-processing the csv file 
	into the RDKit ready form. Generate molecular descriptor,
	normalizing all the input feature by the StandardScaler.

	X is n*m input matrix,
	n is number of data points
	m is number of feature

	y is n*1 target vector
	n is number of data points
	the value in the vector is the electrical conductivity
	'''
	#Data Cleaning, separate value and error in EC column
	data = pd.read_excel("inputdata.xlsx")
	data['EC_value'], data['EC_error'] = 
			zip(*data['ELE_COD'].map(lambda x: x.split('±')))

	#Setting up for molecular descriptors
	n = data.shape[0]
	list_of_descriptors = ['NumHeteroatoms','ExactMolWt','NOCount',
				'NumHDonors','RingCount','NumAromaticRings',
				'NumSaturatedRings','NumAliphaticRings']
	calc = Calculator(list_of_descriptors)
	D = len(list_of_descriptors)
	d = len(list_of_descriptors)*2 + 4
	print(n,d)

	X = np.zeros((n,d))
	X[:,-3] = data['T']
	X[:,-2] = data['P']
	X[:,-1] = data['MOLFRC_A']

	for i in range(n):
    	A = Chem.MolFromSmiles(data['A'][i])
    	B = Chem.MolFromSmiles(data['B'][i])
    	X[i][:D]    = calc.CalcDescriptors(A)
    	X[i][D:2*D] = calc.CalcDescriptors(B)

	new_data = pd.DataFrame(X,columns=['NUM','NumHeteroatoms_A','MolWt_A',
					'NOCount_A','NumHDonors_A','RingCount_A',
					'NumAromaticRings_A','NumSaturatedRings_A',
					'NumAliphaticRings_A','NumHeteroatoms_B',
					'MolWt_B','NOCount_B','NumHDonors_B',
					'RingCount_B','NumAromaticRings_B',
					'NumSaturatedRings_B',
					'NumAliphaticRings_B',
					'T','P','MOLFRC_A'])

	Y = data['EC_value']

	return new_data, Y  

