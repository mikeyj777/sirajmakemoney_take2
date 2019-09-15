import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import OrthogonalMatchingPursuit
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta


df = pd.read_csv('data/KO.csv')

close_px = np.asarray(df['Adj Close'])
close_px = close_px.reshape(-1, 1)
df['Date'] = df['Date'].astype('datetime64[D]')
df['Serial'] = df['Date'] - min(df['Date'])
thedate = np.asarray(df['Serial']) / np.timedelta64(1, 'D')
thedate = thedate.reshape(-1, 1)

Y_train = close_px[:-20]
Y_test = close_px[-20:]

# Split the targets into training/testing sets
X_train = thedate[:-20]
X_test = thedate[-20:]

models = {}


models['lin_model'] = linear_model.LinearRegression()
models['omp'] = OrthogonalMatchingPursuit()
models['bayes_ridge'] = linear_model.BayesianRidge()

preds = {}
variances = {}

plt.scatter(X_test, Y_test,  color='red')

for title, model in models.items():
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    variances[title] = mean_squared_error(Y_test, pred)
    plt.plot(X_test, pred, c=np.random.rand(3,), label=title)


bestmodel = min(variances, key=variances.get)

indate = name = input(
    'Please enter a date after 09-06-2019 (use that date format):  ')
indate = datetime.strptime(indate, '%m-%d-%Y')
datediff = (indate - min(df['Date'])) / timedelta(days=1)

print('predicted stock price: ', models[bestmodel].predict(datediff))

plt.legend()

# model predictions are very close, and show as overlapping on the
# graphical plot.
plt.show()
