import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

degree = 4

class mainFunc():
    def __init__(self):
        self.proc_LoadData()
        self.proc_LinearTraining()
        self.proc_DisplayData()
        
    def proc_LoadData(self):
        self.dataBin = pd.read_csv("csv/canada_per_capita_income.csv")
        print(self.dataBin)
    
    def proc_LinearTraining(self):
        poly_feat = PolynomialFeatures(degree=degree)
        x_poly= poly_feat.fit_transform(self.dataBin[['year']])
        
        self.reg = LinearRegression()
        self.reg.fit(x_poly, self.dataBin['per capita income (US$)'])
        
        self.y_pred = self.reg.predict(x_poly)
        
        print('\ncoef: ', self.reg.coef_)
        print('intercep: ', self.reg.intercept_)
        
        rmse = np.sqrt(mean_squared_error(self.dataBin['per capita income (US$)'], self.y_pred))
        r2 = r2_score(self.dataBin['per capita income (US$)'], self.y_pred)
        print("\nRmse: ", rmse)
        print("\nR^2:  ", r2)
    
    def proc_DisplayData(self):
        plt.plot(self.dataBin['year'], self.dataBin['per capita income (US$)'], marker='x', color='red', alpha=0.4)
        plt.plot(self.dataBin['year'], self.y_pred, color = 'blue', marker = '.',alpha=0.8)
        
        plt.xlabel("year")
        plt.ylabel("per capita income (US$)")
        plt.show()
    
mainFunc()