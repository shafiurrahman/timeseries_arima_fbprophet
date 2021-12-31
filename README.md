# timeseries_arima_fbprophet
timeseries-using-statistical models and facebook prophet

Time series analysis comprises methods for analyzing time series data in order to extract meaningful statistics and other characteristics of the data. Time series forecasting is the use of a model to predict future values based on previously observed values of the same component.


STEP 1 : IMPORTING THE CODE
First, we will load the required packages

STEP 2: MODEL ESTIMATION
Now that we have imported the data as time series, we need to plot the time series data and its ACF PACF. This will help us to have an idea about the time series data and answer the following questions:

The Time Series is stationary or not?
Does the Times Series have constant mean ?
IS there a stochastic or deterministic trend in the data.

STEP 3 : TESTS FOR STATIONARITY
Though it is clearly seen in the plots that the time series is not stationary, we will still perform the adf test , at a significance level of 0.05, to determine the stationarity.

STEP 4: PARAMETER ESTIMATION
Now that we have made the time series stationary, we will fit in the model for the original time series data and mention d=2. The above ACF and PACF suggests that MA(1) is a good candidate.
arma now deprecated, arima, sarima or VAR , fbprophet

STEP 6 : RESIDUAL ANALYSIS
While modeling the time series data, we made certain assumptions of the nature of error.

We need to plot the residual, its ACF and PACF in order to see if the assumptions are still intact. If the model fits the data well, then residual will behave like white noises.

STEP 7 : FORECASTING
Now that we know the model is a good fit, we need to forecast for the next two years. This can be done as follows:
