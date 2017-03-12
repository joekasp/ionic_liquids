import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from methods import lasso, mlp_classifier, mlp_regressor, svr
from sklearn.metrics import mean_squared_error

def make_plot(title, y_test):
		start = y_test.min()
		end = y_test.max()
		xdr = Range1d(start=start, end=end)
		ydr = Range1d(start=start, end=end)
		circles_source = ColumnDataSource(y_test.values.astype(np.float), object.predict(X_test))
    	plot = Plot(
        	x_range=xdr, y_range=ydr, data_sources=circles_source, title=title, 
			width=400, height=400, border_fill='white', background_fill='#e9e0db')
		plot.title.text = title
    	
    	circle = Circle(
        	x=xname, y=yname, size=12,
        	fill_color="#cc6633", line_color="#cc6633", fill_alpha=0.5
    	)
		plot.add_glyph(circles_source, circle)

    	return plot
		#plot = plt.figure(figsize=(4,4))
		#plot.scatter(y_test.values.astype(np.float), object.predict(X_test))
	    #plot.plot([0,12],[0,12],lw=4,c = 'r')
		#return plot 
