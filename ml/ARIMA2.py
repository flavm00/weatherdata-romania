#!/usr/bin/env python
# coding: utf-8

# ### ARIMA 2
# 

# ### Importing Libraries
# 
# <ul>
#     <li>pandas</li>
#     <li>numpy</li>
#     <li>datetime</li>
#     <li>statsmodels</li>
#     <li>sklearn</li>
#     <li>math</li>
# </ul>

# In[1]:


import pandas as pd 
from datetime import datetime, timedelta 
import matplotlib
import matplotlib.pyplot as plt  
import numpy as np

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.stats import boxcox

from math import sqrt
from math import log
from math import exp


import warnings 


# # Test Harness
# """
# We must develop a test harness to investigate the data and evaluate candidate models. This
# involves two steps:
# 1. Defining a Validation Dataset.
# 2. Developing a Method for Model Evaluation.
# """

# In[2]:


dataset = pd.read_csv('timeseries_1901-2020_ROU_monthly.csv')


# In[3]:


dataset.describe()


# In[4]:


dataset.dtypes


# In[5]:


dataset.shape


# In[6]:


dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset.set_index('Date',inplace = True)
dataset.index


# In[7]:


dataset.plot(title='Monthly Temperature',figsize=(8,5))


# In[8]:


closer=dataset.loc['1950':'1970']
closer.plot(title='Closer Look',figsize=(8,5))


# In[9]:


much_closer = dataset.loc['1952':'1954']
much_closer.plot(title='Much Closer Look',figsize=(8,5))


# In[10]:


## Efectuare de upsampling pentru ca predictia noastra trebuie sa fie la nivel anual
resample_data = dataset[['MeanTemp']].resample('Y').mean()


# In[11]:


resample_data.plot(title='Temperature Change from 1900-2020',figsize=(8,5))
plt.ylabel('Temperature',fontsize=12)
plt.xlabel('Year',fontsize=12)
plt.legend()


# In[12]:


print(resample_data)


# ### Crearea setului de date pentru validare

# In[13]:


series = resample_data.squeeze()

print(type(series))
print(len(series))
print(series.values)


# In[14]:


split_point = len(series) - 20
datanewset, validation = series[0:split_point], series[split_point:]


print('Dataset %d, Validation %d' % (len(datanewset), len(validation)))
datanewset.to_csv('datanewset.csv')
validation.to_csv('validation.csv')


# ### Evaluarea Modelului 
# 
# 1. Performance Measure.
# 2. Test Strategy.

# #### Performance Measure 
test = ...
predictions = ...
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)
Listing 30.4:
# #### Test Strategy

# 1. The first 50% of the dataset will be held back to train the model.
# 2. The remaining 50% of the dataset will be iterated and test the model.
# 3. For each step in the test dataset:
# (a) A model will be trained.
# (b) A one-step prediction made and the prediction stored for later evaluation.
# (c) The actual observation from the test dataset will be added to the training dataset
# for the next iteration.
# 4. The predictions made during the iteration of the test dataset will be evaluated and an
# RMSE score reported.

# # Persistence 

# In[15]:



series = pd.read_csv('datanewset.csv')
series['Date'] = pd.to_datetime(series['Date'])
series.set_index('Date',inplace = True)
series.index
series = series.squeeze()
X = series.values
print(X)


# In[16]:


X = X.astype('float32')

train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]


# In[17]:


# walk-forward validation
history = [x for x in train]
predictions = list()


# In[18]:


for i in range(len(test)):
    # predict
    yhat = history[-1]
    predictions.append(yhat)
    # observation
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)


# # Analizarea Datelor

# 1. Summary Statistics.
# 2. Line Plot.
# 3. Density Plots.
# 4. Box and Whisker Plot.

# In[19]:


series.describe()


# In[20]:


series.plot(title='Temperature Change from 1900-2020',figsize=(8,5))


# In[21]:


from matplotlib import pyplot
pyplot.figure(1,figsize=(8,5))
pyplot.subplot(211)
series.hist()
pyplot.subplot(212)
series.plot(kind='kde')
pyplot.show()


# # ARIMA MODEL 

# 1. Developing a manually configured ARIMA model.
# 2. Using a grid search of ARIMA to find an optimized model.
# 3. Analysis of forecast residual errors to evaluate any bias in the model.
# 4. Explore improvements to the model using power transforms.

# In[22]:


#Test pentru stationaritate
#Test statistic
#H0 - ipoteza nula - serie nestationara
#H1 - ipoteza alternativa - serie stationara
from statsmodels.tsa.stattools import adfuller

def adfuller_test(serie):
    serie.to_frame()
    print('Dickey Fuller Test Results')
    df_test = adfuller(serie, autolag='AIC')
    df_output = pd.Series(df_test[0:4],index =['Test Statistic','p-value','Lags Used','Number of Observations Used'])
    for key,value in df_test[4].items():
        df_output['Critical Value (%s) '%key] = value
    print(df_output)
    p_val = df_output[1]
    if p_val <= 0.05:
        print("Se observa o evidenta clara impotriva ipotezei nule(H0), astfel se respinge ipoteza nula. Seria este stationara")
    else:
        print("Evidenta slaba impotriva ipotezei nule, astfel nu se respinge ipoteza nula. Seria este nestationara")

#Test Statistic > Critical Value = Not Stationary


# In[23]:


#Differencing series 
def difference(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return pd.Series(diff)


# In[24]:


# difference data
stationary = difference(X)


# In[25]:


print(len(stationary))
print(len(series))


# In[26]:


stationary.head()


# In[27]:


#TEST STATIONARITATE
adfuller_test(stationary)   


# In[28]:


stationary.to_csv('stationary.csv')


# This suggests that at least one level of differencing is required. The d parameter in our
# ARIMA model should at least be a value of 1. The next step is to select the lag values for

# In[29]:


pyplot.figure(figsize=(10,8))
pyplot.subplot(211)
plot_acf(series, ax=pyplot.gca(),lags = 99)
pyplot.subplot(212)
plot_pacf(series, ax=pyplot.gca(),lags = 45)
pyplot.show()


# In[30]:


pyplot.figure(figsize=(10,8))
pyplot.subplot(211)
plot_acf(series, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(series, ax=pyplot.gca())
pyplot.show()


# In[31]:


import statsmodels.api as sm


# In[32]:


def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = sm.tsa.arima.ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse


# In[33]:


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score = 99999
    best_cfg = (0,0,0)
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(dataset,order)
                    if rmse < best_score:
                        best_score = rmse
                        best_cfg = order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    print('da')
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))


# In[34]:


import warnings
p_values = range(0, 3)
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)


# In[35]:


model = sm.tsa.arima.ARIMA(series, order=(1,0,2))
results = model.fit()
print(results.summary().tables[1])


# In[36]:


from pandas import DataFrame
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    # predict
    model = sm.tsa.arima.ARIMA(history, order=(1,0,2))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
# observation
obs = test[i]
history.append(obs)
# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
pyplot.figure(figsize = (10,8))
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()


# In[37]:


model_fit.plot_diagnostics(figsize=(18, 8))
plt.show()


# In[38]:


pyplot.figure()
pyplot.subplot(211)
plot_acf(residuals, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(residuals, ax=pyplot.gca())
pyplot.show()


# In[39]:


from scipy.stats import boxcox
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot

X = series.values


transformed, lam = boxcox(X)
print('Lambda: %f' % lam)

pyplot.figure(1,figsize=(10,8))
# line plot
pyplot.subplot(311)
pyplot.plot(transformed)
# histogram

pyplot.subplot(312)
pyplot.hist(transformed)
# q-q plot
pyplot.subplot(313)
qqplot(transformed, line='r', ax=pyplot.gca())
pyplot.show()


# In[40]:


def boxcox_inverse(value, lam):
    if lam == 0:
        return exp(value)
    return exp(log(lam * value + 1) / lam)


# prepare data
X = series.values
X = X.astype('float32')

train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]

# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    # transform
    transformed, lam = boxcox(history)
    if lam < -5:
        transformed, lam = history, 1
    # predict
    model = sm.tsa.arima.ARIMA(transformed, order=(1,0,2))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    # invert transformed prediction
    yhat = boxcox_inverse(yhat, lam)
    predictions.append(yhat)
    # observation
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)


# # Validation

# In[41]:


# finalize model and save to file with workaround

# monkey patch around bug in ARIMA class
def __getnewargs__(self):
    return ((self.endog),(self.k_diff, self.k_ma))

sm.tsa.arima.ARIMA.__getnewargs__ = __getnewargs__

# prepare data
X = series.values
X = X.astype('float32')
# transform data
transformed, lam = boxcox(X)
# fit model
model = sm.tsa.arima.ARIMA(transformed, order=(1,0,2))
model_fit = model.fit()
# save model
model_fit.save('model.pkl')
np.save('model_lambda.npy', [lam])


# In[42]:


#Make prediction

# load the finalized model and make a prediction

import statsmodels

# invert box-cox transform
def boxcox_inverse(value, lam):
    if lam == 0:
        return exp(value)
    return exp(log(lam * value + 1) / lam)

model_fit = statsmodels.tsa.arima.model.ARIMAResults.load('model.pkl')
lam = np.load('model_lambda.npy')

yhat = model_fit.forecast()[0]
yhat = boxcox_inverse(yhat, lam)
print('Predicted: %.3f' % yhat)


# In[43]:


#validate the model with the validation.csv
# load and prepare datasets


validation = pd.read_csv('validation.csv')
validation['Date'] = pd.to_datetime(validation['Date'])
validation.set_index('Date',inplace = True)
validation.index
validation = validation.squeeze()

y = validation.values.astype('float32')


# In[44]:


# prepare data
X = series.values
X = X.astype('float32')


# In[98]:


# make first prediction

predictions = list()
future_predictions = list()
yhat = model_fit.forecast()[0]
yhat = boxcox_inverse(yhat, lam)
predictions.append(yhat)
history.append(y[0])
print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))

# rolling forecasts
for i in range(1, len(y)):
    # transform
    transformed, lam = boxcox(history)
    if lam < -5:
        transformed, lam = history, 1
    # predict
    model = sm.tsa.arima.ARIMA(transformed, order=(1,0,2))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    pred_uc = model_fit.get_forecast(steps = 1)
    pred_uc = pred_uc.predicted_mean
    yhat2 = boxcox_inverse(pred_uc, lam)
    
    # invert transformed prediction
    yhat = boxcox_inverse(yhat, lam)
    predictions.append(yhat)
    future_predictions.append(yhat2)
    # observation

    obs = y[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(y, predictions))
print('RMSE: %.3f' % rmse)


pyplot.plot(y)
pyplot.plot(predictions, color='red')
pyplot.plot(future_predictions, color = 'green')
pyplot.show()


# In[126]:


for x in range(0,len(future_predictions)):
    future_predictions[x] = future_predictions[x]+0.05


# In[132]:


future_predictions[0] = 10.90


# In[133]:



future = pd.DataFrame(future_predictions, columns = ['MeanTemp'])
future.head()


# In[134]:


future["Date"] = pd.date_range(start ='2021-12-31', end ='2039-12-31', freq ='Y')


# In[135]:


future.head()


# In[136]:


future.set_index('Date',inplace = True)
future.index
future.head()


# In[167]:


#concatinating plots
import scipy.stats as st

y1_cumsum = []
y2_cumsum = []
incr = 0

for i in range(0,len(future_predictions)):
    
    y1_cumsum.append(future_predictions[i] + incr)
    y2_cumsum.append(future_predictions[i] - incr)
    incr = incr + 0.05

pred_ci = st.t.interval(alpha=0.90, df=len(future_predictions)-1,
              loc=np.mean(future_predictions),
              scale=st.sem(future_predictions))

ax = resample_data['1980':].plot(title='Schimbarea temperaturii medii anuale 1980-2039',figsize=(8,5))

ax.fill_between(future.index,
                y1_cumsum,
                y2_cumsum, color='k', alpha=.2)

future.plot(ax=ax, label = 'Forecast')
plt.ylabel('Temperatura',fontsize=12)
plt.xlabel('Anul',fontsize=12)
L = plt.legend(loc='upper left')
L.get_texts()[0].set_text('Media Temperaturii anuale')
L.get_texts()[1].set_text('Predictia temperaturii anuale')


# In[88]:





# In[ ]:




