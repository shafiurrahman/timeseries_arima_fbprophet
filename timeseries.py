import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

df=pd.read_csv("air_pollution.csv",index_col='date',parse_dates=True)
df.head()

df.columns
df.isnull().sum()
df.tail()
df.index.freq="D"
df.index
df['new']=df['pollution_today'].plot(legend=True)

from statsmodels.tsa.seasonal import seasonal_decompose
result=seasonal_decompose(x=df['pollution_today'],model='additive')
result

result.plot();

#moving average
mov_avg=df['pollution_today'].ewm(span=12).mean()
mov_avg
mov_avg.plot()
df['6month_mavg']=df['pollution_today'].rolling(window=180).mean()
df['6month_mavg'].plot()
df[['pollution_today','6month_mavg']].plot(legend=True)
# further other methods too here
import statsmodels.api as sm
from statsmodels.tsa.stattools import acovf,acf,pacf,pacf_yw,pacf_ols
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pandas.plotting import lag_plot
lag_plot(df['pollution_today'])
plot_acf(x=df['pollution_today'],lags=40);
plot_pacf(x=df['pollution_today'],lags=40);

#differrencing
from statsmodels.tsa.statespace.tools import diff
df['first_order']=diff(series=df['pollution_today'],k_diff=1).plot(legend=True)
df['third_order']=diff(series=df['pollution_today'],k_diff=3).plot(legend=True)

#holt winters method
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
span=365  # 12 months
alpha=2/(span+1)
print(alpha)
df['pollution_today'].ewm(alpha= alpha ,adjust=False).mean()
df['sse']=SimpleExpSmoothing(df['pollution_today']).fit(smoothing_level=alpha,optimized=False).fittedvalues.shift(-1)
df['sse'].plot()
df['doublesse']=ExponentialSmoothing(endog=df['pollution_today'],trend='add').fit().fittedvalues.shift(-1).plot()
#we use extra layer seasonal in ExponentialSmoothing()--this is holts walter model
df['TESadd12']=ExponentialSmoothing(endog=df['pollution_today'],trend='add',seasonal='add',seasonal_periods=4).fit().fittedvalues
df[['TESadd12','doublesse','sse']].plot(legend=True)

from statsmodels.tsa.arima_model import ARMA,ARIMA,ARMAResults,ARIMAResults
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
   
def adf_test(series,title=''):
   
    #Pass in a time series and an optional title, returns an ADF report
        
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data\n",
        
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)
    
    for key,val in result[4].items():
        out[f'critical value ({key})']=val
    print(out.to_string())          # .to_string() removes the line \"dtype: float64\"\n",
        
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis"),
        print("Data has a unit root and is non-stationary")
   
adf_test(df["pollution_today"])
auto_arima(df["pollution_today"],seasonal=False).summary()
train=df.iloc[:1460]
test=df.iloc[1460:]

len(df)
import statsmodels
test.mean()

##arima
stepwise_fit=auto_arima(df['pollution_today'],start_p=0, start_q=0, max_p=6, max_q=3,seasonal=False,Trace=True,
                        information_criterion='aic',stepwise=False,m=365)
stepwise_fit
stepwise_fit.summary()
from statsmodels.tsa.arima_model import ARMA,ARIMA,ARMAResults,ARIMAResults
model=statsmodels.tsa.arima.model.ARIMA(endog=train['pollution_today'],order=(2,1,1))
results=model.fit()
results.summary()
start=len(train)
end=start+len(test)-1
predictions1=results.predict(start=start,end=end,typ='levels').rename("arima(2,1,1) predictions ")
predictions1

test
test['pollution_today'].plot(legend=True)
predictions1.plot(legend=True)
#evaluting the model
from statsmodels.tools.eval_measures import rmse
error=rmse(test['pollution_today'],predictions1)
error
test['pollution_today'].mean()
#forecast into unknown future
model=statsmodels.tsa.arima.model.ARIMA(df['pollution_today'],order=(2,1,1))
results=model.fit()
forecast=results.predict(start=len(df),end=len(df)+365,typ='levels').rename("arima(2,1,1) forecast ")
df["pollution_today"].plot(legend=True)
forecast.plot(legend=True)

#sarimax
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima

model=SARIMAX(df['pollution_today'],order=(2,1,1),seasonal_order=(1,0,1,12))
result=seasonal_decompose(x=df['pollution_today'],model='additive')
result.plot();
results=model.fit()
results.summary()

start=len(df)
end=start+len(test)-1
predictions=results.predict(start,end,typ='levels').rename('SARIMA Predictions')
predictions
df['pollution_today'].plot(legend=True)
predictions.plot(legend=True)
error=rmse(test['pollution_today'],predictions)
error # 
test['pollution_today'].mean()
#forecast into the unknown future

model=SARIMAX(df['pollution_today'],order=(2,1,1),seasonal_order=(1,0,1,12))
result=model.fit()
fcast=result.predict(len(df),len(df)+11,typ='levels').rename("forecasted sarimax")
test['pollution_today'].plot(legend=True)
fcast.plot(legend=True)

###using FACEBOOK PROPHET
import pandas as pd
from fbprophet import Prophet
fdf=pd.read_csv("air_pollution.csv")
fdf.head()
### Load Data

#The input to Prophet is always a dataframe with two 
# columns: ds and y. The ds (datestamp) column should be of a 
# format expected by Pandas, ideally YYYY-MM-DD for a date or
#  YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be
#  numeric, and represents the measurement we wish to forecast.

fdf=fdf[["date","pollution_today"]]
fdf.head()
#Format the Data
fdf.columns = ['ds','y']
fdf.columns
fdf['ds'] = pd.to_datetime(fdf['ds'])
fdf['ds']

#Create and Fit Model
# This is fitting on all the data (no train test split in this example)
m = Prophet()
m.fit(fdf)

#Forecasting
#Step 1: Create "future" placeholder dataframe
#NOTE: Prophet by default is for daily data. You need to pass a frequency for sub-daily or monthly data. Info: https://facebook.github.io/prophet/docs/non-daily_data.html

future = m.make_future_dataframe(periods=24,freq = 'MS')
fdf.tail()
len(fdf)
len(future)
#Step 2: Predict and fill in the Future
forecast = m.predict(future)
forecast.head()
forecast.tail()
forecast.columns
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)

#Plotting Forecast
#We can use Prophet's own built in plotting tools
m.plot(forecast);
import matplotlib.pyplot as plt
import datetime
m.plot(forecast)
#plt.xlim('2014-01-01','2022-01-01')
plt.xlim([datetime.date(2014, 1, 1), datetime.date(2021, 1, 1)])
forecast.plot(x='ds',y='yhat')
m.plot_components(forecast);

#Forecast Evaluations
fdf.info()
len(fdf)
len(fdf) - 12
train = fdf.iloc[:1813]
test = fdf.iloc[1813:]
m = Prophet()
m.fit(train)
future = m.make_future_dataframe(periods=12,freq='MS')
forecast = m.predict(future)
forecast.tail()
test.tail()
ax = forecast.plot(x='ds',y='yhat',label='Predictions',legend=True,figsize=(8,8))

test.plot(x='ds',y='y',label='today_pollution',legend=True,ax=ax,xlim=('2011-01-01','2016-01-01'))
#we are predicting 2015
from statsmodels.tools.eval_measures import rmse
predictions = forecast.iloc[-12:]['yhat']
predictions
test['y']
rmse(predictions,test['y'])
test.mean()

###Prophet Diagnostics
#Prophet includes functionality for time series cross validation to measure forecast error using historical data. This is done by selecting cutoff points in the history, and for each of them fitting the model using data only up to that cutoff point. We can then compare the forecasted values to the actual values.
from fbprophet.diagnostics import cross_validation,performance_metrics
from fbprophet.plot import plot_cross_validation_metric

len(fdf)
len(fdf)/12
#The initial period should be long enough to capture all of the components of the model, in particular seasonalities and extra regressors: at least a year for yearly seasonality, at least a week for weekly seasonality, etc.
# help(pd.Timedelta)
# Initial2 years training period
initial = 3 * 365
initial = str(initial) + ' days'
# Fold every 5 years
period = 3 * 365
period = str(period) + ' days'
# Forecast 1 year into the future
horizon = 365
horizon = str(horizon) + ' days'
df_cv = cross_validation(m, initial=initial, period=period, horizon = horizon)
df_cv.head()
df_cv.tail()
performance_metrics(df_cv)
plot_cross_validation_metric(df_cv, metric='rmse');
plot_cross_validation_metric(df_cv, metric='mape');
#trend changes
fdf.plot(x='ds',y='y',figsize=(12,8),label='pollution')
m = Prophet()
m.fit(train)
future = m.make_future_dataframe(periods=12,freq='MS')
forecast = m.predict(future)
from fbprophet.plot import add_changepoints_to_plot
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)
