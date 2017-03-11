import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator as Calculator

def molecular_descriptors(data):
	#Setting up for molecular descriptors
	n = data.shape[0]
	list_of_descriptors = ['NumHeteroatoms', 'ExactMolWt', 'NOCount', 'NumHDonors',
		'RingCount', 'NumAromaticRings', 'NumSaturatedRings','NumAliphaticRings']
	calc = Calculator(list_of_descriptors)
	D = len(list_of_descriptors)
	d = len(list_of_descriptors)*2 + 4
	
	Y = data['EC_value']
	X = np.zeros((n,d))
	X[:,-3] = data['T']
	X[:,-2] = data['P']
	X[:,-1] = data['MOLFRC_A']

	for i in range(n):
		A = Chem.MolFromSmiles(data['A'][i])
		B = Chem.MolFromSmiles(data['B'][i])
		X[i][:D]    = calc.CalcDescriptors(A)
		X[i][D:2*D] = calc.CalcDescriptors(B)

	new_data = pd.DataFrame(X,columns=['NUM', 'NumHeteroatoms_A', 'MolWt_A', 'NOCount_A',
		'NumHDonors_A', 'RingCount_A', 'NumAromaticRings_A', 'NumSaturatedRings_A',
		'NumAliphaticRings_A', 'NumHeteroatoms_B', 'MolWt_B', 'NOCount_B', 'NumHDonors_B',
		'RingCount_B', 'NumAromaticRings_B', 'NumSaturatedRings_B', 'NumAliphaticRings_B',
		'T', 'P', 'MOLFRC_A'])
	return new_data, Y


def read_data(filename):
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
	#data['EC_value'], data['EC_error'] = zip(*data['ELE_COD'].map(lambda x: x.split('±')))

    return df


def save_model(obj,model_type,filename='default'):
    items = []
    if (model_type == 'lasso'):
        items.append(obj.coef_)
        items.append(obj.sparse_coef_)
        items.append(obj.intercept_)
        items.append(obj.n_iter_)
    elif (model_type == 'mlp_reg'):
        items.append(obj.loss_)
        items.append(obj.coefs_)
        items.append(obj.intercepts_)
        items.append(obj.n_iter_)
        items.append(obj.n_layers_)
        items.append(obj.n_outputs_)
        items.append(obj.out_activation_)
    elif (model_type == 'mlp_clas'):
        items.append(obj.classes_)
        items.append(obj.loss_)
        items.append(obj.coefs_)
        items.append(obj.intercepts_)
        items.append(obj.n_iter_)
        items.append(obj.n_layers_)
        items.append(obj.n_outputs_)
        items.append(obj.out_activation_)
    elif (model_type == 'svr'):
        items.append(obj.support_)
        items.append(obj.support_vectors_)
        items.append(obj.dual_coef_)
        items.append(obj.coef_)
        items.append(obj.intercept_)
        items.append(obj.sample_weight_)
    else:
        raise ValueError('Invalid model type!')

    if (filename == 'default'):
        filename = 'model' + model_type + '.txt'

    f = open(filename,'w')
    f.write(model_type + '\n')
    for item in items:
        f.write(item)
        f.write('\n')
    f.close()
 
    return


def read_model(filename,model_type):
    if (model_type == 'lasso'):
        obj = Lasso
    elif (model_type == 'mlp_reg'):
        obj = MLPRegressor
    elif (model_type == 'mlp_clas'):
        obj = MLPClassifier
    elif (model_type == 'svr'):
        obj = SVR
    else:
        raise ValueError('Invalid model type!')
    return obj

