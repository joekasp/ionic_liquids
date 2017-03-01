import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator as Calculator

#Hard coded for 1-butyl-3-methylimidazolium methanesulfonate and water
#components abcd, where a and b are cation and anion of one component and
#c and d are cation and anion of another component. 
x = Chem.MolFromSmiles('')



#molecular descriptors such as the number of atoms
num_atoms = m.GetNumAtoms()




#Neural network





