from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
#import requests
#import sqlite3

#conn = sqlite3.connect('lightfild.db')

#c = conn.cursor()

#from flask import Flask
#@app.route("/",methods=['GET'])
#app = Flask(__name__)
# if (__name__=="__main__")
# app.run()
     
#def Forecasting() :
#


def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
# series = c.execute("""SELECT * FROM sales()""")

series = read_csv('Testf.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
series.index = series.index.to_period('M')
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# save model
#predictions.save('predictions.pkl')
#test.save('test.pkl')
# load mode
#loaded = ARIMAResults.load('model.plkl')
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()