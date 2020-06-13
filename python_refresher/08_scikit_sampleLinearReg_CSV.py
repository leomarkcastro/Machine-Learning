import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

class mainFunc():
    def __init__(self):
        self.proc_LoadData()
        self.proc_LinearTraining()
        self.proc_DisplayData()
        
    def proc_LoadData(self):
        self.dataBin = pd.read_csv("csv/canada_per_capita_income.csv")
        print(self.dataBin)
    
    def proc_LinearTraining(self):
        self.reg = linear_model.LinearRegression()
        self.reg.fit(self.dataBin[['year']], self.dataBin['per capita income (US$)'])
        
        print('coef: ', self.reg.coef_)
        print('intercep: ', self.reg.intercept_)
    
    def proc_DisplayData(self):
        plt.plot(self.dataBin['year'], self.dataBin['per capita income (US$)'], marker='x', color='red', alpha=0.4)
        plt.plot(self.dataBin['year'], self.reg.predict(self.dataBin[['year']]), color = 'blue', alpha=0.8)
        
        plt.xlabel("year")
        plt.ylabel("per capita income (US$)")
        plt.show()
    
mainFunc()