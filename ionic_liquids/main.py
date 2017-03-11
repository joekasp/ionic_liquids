import numpy as np
import pandas as pd
from front_functions import * 
import utils

#get X matrix and response vector y (need a function for this) 
X,y = gen_data(filename) 

#do machine_learning call
obj = MLP_regressor(X,y)

#save model to file
utils.save_model(obj,model_type='mlp_reg')


