import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import *
from methods import lasso, mlp_classifier, mlp_regressor, svr, errors

def errors(y_test,y_prediction):
		difference = y_prediction - y_test
		return difference

