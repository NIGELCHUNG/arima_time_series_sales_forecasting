
​import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

print('[1] read excel')
sales_df = pd.read_excel("Sales_20190606_B.xlsx")
print(sales_df)

print('[2] sort by invoice_date')
sales_df = sales_df.sort_values('invoice_date')
print(sales_df)

print('[3] inspect min/max invoice date')
print('min date: ', sales_df['invoice_date'].min())
print('max date: ', sales_df['invoice_date'].max())

print('[4] fix missing data')
sales_df.isnull().sum()
print(sales_df)

# aggregate by invoice date
print('[5] sum rows by invoice date')
sales_df = sales_df.groupby('invoice_date')['total_sales'].sum
().reset_index()
print(sales_df)

# indexing w/ time series data
print('[6] make index invoice_date')
sales_df = sales_df.set_index('invoice_date')
sales_df.index
print(sales_df.index)


print('[7] get mean daily sales for start each month')
y = sales_df['total_sales'].resample('MS').mean()
print(y)

print('[8] plot mean daily sales for each month')
# sales_dfx['2018':]

y.plot(figsize=(15, 6))
plt.show()

print('[9] trend, seasonlity and noise')
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

print('[10] Time series forecasting with ARIMA')
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p,
d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


print('[11]  ')
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal,
results.aic))
        except:
            continue


print('[12] Fitting the ARIMA model')
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

print('[13] run model diagnostics to see any unusual behavior')
results.plot_diagnostics(figsize=(16, 8))
plt.show()

print('[13] Validate forecasts, blue line = observed, orange line =
forecast')
pred = results.get_prediction(start=pd.to_datetime('2017-01-01'),
dynamic=False)
pred_ci = pred.conf_int()
ax = y['2009':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7,
figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Avg Daily Sales for 4 DCs')
plt.legend()
plt.show()


print('[14] determine accuracy')
y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse,
2)))

print('The Root Mean Squared Error of our forecasts is {}'.format(round
(np.sqrt(mse), 2)))

print('[15] Producing and visualizing forecasts')
pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Avg Daily Sales for 4 DCs')
plt.legend()
plt.show()