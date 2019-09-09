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

avetime = 60


def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


df = pd.read_csv('data/KO.csv')
close_px = df['Adj Close']

close_px = np.asarray(df['Adj Close'])
close_px = close_px.reshape(-1, 1)
df['Date'] = df['Date'].astype('datetime64[D]')
df['Serial'] = df['Date'] - min(df['Date'])
thedate = np.asarray(df['Serial']) / np.timedelta64(1, 'D')
thedate = thedate.reshape(-1, 1)

mavg = moving_average(np.asarray(close_px), avetime)
mavg = mavg[~np.isnan(mavg)]
mavg = mavg.reshape(-1, 1)

mavg_date = moving_average(np.asarray(thedate), avetime)
mavg_date = mavg_date.reshape(-1, 1)


regrmavg = linear_model.LinearRegression()
regromp = OrthogonalMatchingPursuit()
regrbayridge = linear_model.BayesianRidge()

# for the studied stock ticker (KO = Coca Cola), the moving average and
# original date give roughly the same results when predicting future
# values.  Leaving the statement below commented out for future testing.

# mavg = close_px
# mavg_date = thedate

regrmavg.fit(mavg_date, mavg)
regromp.fit(mavg_date, mavg)
regrbayridge.fit(mavg_date, mavg)

mavg_pred = regrmavg.predict(mavg_date)
omp_pred = regromp.predict(mavg_date)
bayridge_pred = regrbayridge.predict(mavg_date)

indate = name = input(
    'Please enter a date after 09-06-2019 (use that date format):  ')
indate = datetime.strptime(indate, '%m-%d-%Y')
datediff = (indate - min(df['Date'])) / timedelta(days=1)

preds = []
preds.append(['Ordinary Least Squares (OLS) pred:  ',
              regrmavg.predict(datediff)])
preds.append(['Orthogonal Matching Pursuit (OMP) pred:  ',
              regromp.predict(datediff)])
preds.append(['Bayesian Ridge Regression pred:  ',
              regrbayridge.predict(datediff)])

for pred in preds:
    print(pred[0], pred[1])

plt.scatter(mavg_date, mavg,  color='red')
plt.plot(mavg_date, mavg_pred, color='orange', linewidth=3, label='OLS')
plt.plot(mavg_date, omp_pred, color='blue', linewidth=3, label='OMP')
plt.plot(mavg_date, bayridge_pred, color='green',
         linewidth=3, label='Bayesian Ridge')

plt.legend()

# model predictions are very close, and show as overlapping on the
# graphical plot.
plt.show()
