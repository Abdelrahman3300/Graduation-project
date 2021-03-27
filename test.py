from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas import DataFrame
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import requests
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults

#import sqlite3

#conn = sqlite3.connect('lightfild.db')

#c = conn.cursor()
#from flask import Flask
#@app.route("/",methods=['GET'])
#app = Flask(__name__)
# if (__name__=="__main__")
# app.run()
     
#def anaylsis() :

series = read_csv('Test S.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
print(series.head())
series.plot()
pyplot.show()
autocorrelation_plot(series)
pyplot.show()
model = ARIMA(series['Qty'], order=(1,1,0),exog=series['total'])
model_fit = model.fit()
# save model
#model_fit.save('model.pkl')
# load mode
#loaded = ARIMAResults.load('model.plkl')
print(model_fit.summary())
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

