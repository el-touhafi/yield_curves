from countryinfo import CountryInfo
import requests 
from bs4 import BeautifulSoup as bs
import numpy as np
import re
import pandas as pd
from playwright.sync_api import sync_playwright
from multiprocessing import Pool,cpu_count,freeze_support
from datetime import datetime,timedelta
import time
from io import StringIO
from tvDatafeed import TvDatafeed, Interval
from scipy.optimize import minimize,LinearConstraint
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from Yield_Curves.ModelingDataScraper import ModelingDataScraper


class ModelisationGARCH(ModelingDataScraper):
    """
    The class will serve as a modeler for short-term interest rates with a 1-month maturity.
    Using historical data, we will fit the data to the Vasicek model.
    We will discretize the derivatives of the Vasicek model to determine
    an autoregressive model based on the Vasicek discretization equation.
    Subsequently, a GARCH model will be applied as a conditional
    filtration based on the previous day's data.
    The GARCH(1,1) model will include one factor for lagged volatility and one factor for lagged error.
    This approach will resemble the Heston model applied to interest rates within the Vasicek framework,
    utilizing discretization and capturing both the interest rate and its volatility
    through an autoregressive and GARCH model.
    To use the class give dates and country and as output variable give on of the above:
    ['fit','simulation','expected']
    """
    def __init__(self, pays: str, start_date: str, end_date: str,output :str):
        super().__init__(pays, start_date, end_date)
        self._vasicek_garch_fit = pd.DataFrame()
        self._vasicek_garch_simulation = pd.DataFrame()
        self._vas_garch_monte_carlo_expected = pd.DataFrame()
        self.output = output


    def compute(self):
        """
        The method will be employed to compute the dataset of the fitted model
        and make one-year predictions, whether through simulation or expected values.
        """
        super().compute()
        self.vasicek()
        self.vasicek_garch()
        if self.output == 'fit':
            return self.vasicek_garch_fit
        elif self.output == 'simulation':
            return self.vasicek_garch_simulation
        else:
            return self.vas_garch_monte_carlo_expected

    def vasicek_mle(self, vasicek_params):
        """
        We will proceed under the assumption that interest rates
        follow a normally distributed pattern,
        although this is not universally proven and may only hold true in certain samples.
        Market conditions can vary significantly, leading to substantial fluctuations and
        occasional crisis scenarios, which may deviate from normality.
        Therefore, the model may not be suitable for stress testing.
        Nevertheless, we will employ maximum likelihood estimation based on a normal distribution
        to optimize the Vasicek model's factors for improved fitting to historical data.
        equation vasicek : dr(t) = alpha * (theta - r(t)) dt + sigma * dW(t)
        autorogressive model : r(t) = r(t-1) + alpha * (theta - r(t-1)) dt + sigma * de(t-1)
        e white noise.
        """
        data_copy = self.model_data.iloc[:,-1]
        deterministic_equation = np.array(vasicek_params[0] * vasicek_params[1] + vasicek_params[0] * np.array(data_copy.shift(1).dropna()))
        discretization_derivative = np.array((data_copy.shift(-1)-data_copy).dropna())
        error = sum((discretization_derivative-deterministic_equation)**2)
        return error
    def vasicek(self):
        """
        The BFGS method is used for approximating the parameters of nelson siegel formula
        check out the method it's called also quasi-Newton's Method wich consist on
        Hessian matrix.
        """
        initial_params = [0.1,0.1]
        optimal_params = minimize(self.vasicek_mle,initial_params,method='BFGS',tol=1e-100)
        optimal_params.x = np.absolute(optimal_params.x)
        self.vasicek_params =  optimal_params.x
    def vasicek_garch_mle(self, C):
        """
        Additionally, after estimating the Vasicek parameters,
        we will utilize maximum likelihood optimization for the volatility in the GARCH(1,1) model.
        Equation : sigma**2(t) = factor1 * sigma**2(t-1) + factor2 * error**2(t-1) + omega
        full equation : r(t) = r(t-1) + alpha * (theta - r(t-1)) dt + sigma(t) * de(t-1)
        """
        params_copy = self.vasicek_params
        data_copy = self.model_data.iloc[:,-1]
        fit_data = np.array(data_copy.shift(1).dropna()) + params_copy[0]*params_copy[1] - params_copy[0]*np.array(data_copy.shift(1).dropna())
        real_data = np.array((data_copy.shift(-1)).dropna())
        error = np.abs(real_data-fit_data)
        estimated_vol = np.zeros(len(data_copy)-1)
        estimated_vol[0] = (C[0]/(1-C[1]-C[2]))**(1/2)
        for i in range(1,len(self.model_data.index)-1):
            estimated_vol[i]=(C[0]+C[1]*error[i-1]**2+C[2]*estimated_vol[-1]**2)**(1/2)
        likelihood_normal_dist = 1/((2*math.pi)**(1/2)*estimated_vol) * np.exp(-error**2/(2*estimated_vol**2))
        likelihood_log_sum = sum(np.log(likelihood_normal_dist))
        return -likelihood_log_sum


    def vasicek_garch(self):
        """
        The BFGS method is used for approximating the parameters of nelson siegel formula
        check out the method it's called also quasi-Newton's Method wich consist on
        Hessian matrix.
        """
        data_copy = self.model_data.iloc[:,-1]
        volatility = float(data_copy.std())
        intial_params = [volatility,0,0]
        optimal_params = minimize(self.vasicek_garch_mle, intial_params, bounds=[(0, None), (0, None), (0, None)], constraints=({'type': 'eq', 'fun': lambda C: C[0] / (1 - C[1] - C[2])}))
        self.garch_params = optimal_params.x

    @property
    def vasicek_garch_fit(self):
        """
        The Vasicek-GARCH(1,1) model fit method to historical data.
        """
        vasicek_params = self.vasicek_params
        data_copy = self.model_data.iloc[:,-1]
        vasicek_fit = np.array(data_copy.shift(1).dropna()) + vasicek_params[0]*vasicek_params[1] - vasicek_params[0]*np.array(data_copy.shift(1).dropna())
        real_data = np.array((data_copy.shift(-1)).dropna())
        garch_params = self.garch_params
        vol_garch = np.zeros(len(data_copy)-1)
        vol_garch[0] = (garch_params[0]/(1-garch_params[1]-garch_params[2]))**(1/2)
        vasicek_garch_fit = [data_copy.iloc[0]]
        error = np.abs(real_data-vasicek_fit)
        for i in range(1,len(self.model_data.index)-1):
            brownien_motion = np.random.normal(0, 1)
            vol_garch[i]=(garch_params[0]+garch_params[1]*error[i-1]**2+garch_params[2]*vol_garch[-1]**2)**(1/2)
            vasicek_garch_fit.append(vasicek_garch_fit[-1]*np.exp(-vasicek_params[0])+vasicek_params[1]*(1-np.exp(-vasicek_params[0]))+vol_garch[i]*brownien_motion)
        self._vasicek_garch_fit = pd.DataFrame(vasicek_garch_fit,index=self.model_data.index[:-1])
        return self._vasicek_garch_fit

    @property
    def vasicek_garch_simulation(self):
        """
        The Vasicek-GARCH(1,1) model simulation method to predict 360 days.
        """
        vasicek_params = self.vasicek_params
        garch_params = self.garch_params
        fit_data = (np.array(self.model_data.iloc[-2,-1]) + vasicek_params[0] * vasicek_params[1] - vasicek_params[0] * np.array(self.model_data.iloc[-2,-1]))
        real_data = np.array(self.model_data.iloc[-1,-1])
        volatility = [np.abs(real_data-fit_data)]
        simulated_data = [self.model_data.iloc[-2,-1]]
        for i in range(360):
            brownien_motion = np.random.normal(0,1,size=360)
            simulated_data.append(simulated_data[-1] + self.vasicek_params[0] * (self.vasicek_params[1] - simulated_data[-1]) + volatility[-1] * brownien_motion[i])
            volatility.append((garch_params[0]+garch_params[1]*(simulated_data[-1]-simulated_data[-2])**2+garch_params[2]*volatility[-1]**2)**(1/2))
        self._vasicek_garch_simulation = pd.DataFrame(simulated_data,index=np.array(pd.date_range(start= self.model_data.index[-2], end=(pd.to_datetime(self.model_data.index[-2]) + pd.to_timedelta(360, unit='D')), freq='D')))
        return self._vasicek_garch_simulation

    def vas_garch_monte_carlo_simulation(self,simulations_number):
        """
        The Vasicek-GARCH(1,1) model Monte Carlo Simulation method to predict
        360 days based on simulation number.
        """
        many_simulated_data = self.vasicek_garch_simulation
        for i in range(simulations_number):
            many_simulated_data = pd.concat([many_simulated_data,self.vasicek_garch_simulation],axis=1)
        return many_simulated_data

    @property
    def vas_garch_monte_carlo_expected(self):
        """
        Expected value-based Monte Carlo simulation allows you to adjust
        the number of simulations to obtain the expected value over 360 days.
        This approach will provide the autoregressive Vasicek function as a mean
        without incorporating volatility or white noise.
        """
        many_simulated_data = self.vas_garch_monte_carlo_simulation(1000)
        self._vas_garch_monte_carlo_expected = many_simulated_data.mean(axis=1)
        return self._vas_garch_monte_carlo_expected


    
b = 'BR'
p = ModelisationGARCH(b, '01-01-2000','25-08-2024','simulation')
p.compute()
df = p.vas_garch_monte_carlo_simulation(300)
print(df)

plt.plot(pd.to_datetime(p.model_data.index[4000:]),p.model_data.iloc[4000:,1])
plt.plot(p.vas_garch_monte_carlo_expected.index,p.vas_garch_monte_carlo_expected)
for comp, i in enumerate(df.columns):
    plt.plot(df.index, df.iloc[:, comp])
"""
b = 'MA'            
d = ModelisationGARCH(b, '01-01-2013','01-01-2023')
#f = ModelisationAR(b, '01-01-2013','01-01-2023')
l= np.array(d.CIR())
l1 = np.array(d.VASICEK())
#L,a = f.VASICEK('MLE')
#L1,a1 = f.CIR('MLE')
K = np.array(l+l1)/2#np.array(L[1:])+np.array(L1[1:])+


x = np.arange(0,len(d.data.index))
#plt.plot(x[1:],K,c = 'pink',label='mean of the four '+d.CC)
#plt.plot(x[1:],l,c = 'green',label='CIR GARCH '+d.CC)
plt.plot(x[1:],l1,c = 'yellow',label='VASICEK GARCH '+d.CC)
#plt.plot(x,L,c = 'red',label='Vasicek MLE '+d.CC)
#plt.plot(x,L1,c = 'purple',label='CIR MLE '+d.CC)
plt.plot(x,d.data.iloc[:,-1],c = 'black',label='Real Data of '+d.CC)
plt.title('modeling short interest rate for '+d.CC)
plt.legend()
plt.tick_params(axis='x',which='both', bottom=False,  top=False, labelbottom=False)
plt.xticks(ticks=None,rotation=45)
plt.show()
"""
