import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class mainFunc():
    def __init__(self):
        self.proc_LoadData()
        self.proc_TreatData()
        self.proc_TrainModel()
        #self.proc_GraphModel()
    
    def proc_LoadData(self):
        self.dataBin = pd.read_csv("csv/carprices.csv") 
        self.X = self.dataBin[['Mileage','Age(yrs)']]
        self.y = self.dataBin[['Sell Price($)']]
    
    def proc_TreatData(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2)
    
    def proc_DisplayData(self):
        pass
    
    def proc_TrainModel(self):
        self.linearModel = linear_model.LinearRegression()
        self.linearModel.fit(self.X_train, self.y_train)
        
        print("Accuracy: ", self.linearModel.score(self.X_test, self.y_test))
        print("coef:", self.linearModel.coef_)
        print("intercept: ", self.linearModel.intercept_)
        
    def proc_GraphModel(self):
        
        m = self.linearModel.coef_[0,0]
        b = self.linearModel.intercept_
        plt.plot(self.dataBin['Mileage'], [(m*i + b) for i in self.dataBin['Mileage']], color="red")
        
        plt.show()

        
        m = self.linearModel.coef_[0,1]
        b = self.linearModel.intercept_
        plt.plot(self.dataBin['Age(yrs)'], [(m*i + b) for i in self.dataBin['Age(yrs)']], color="blue")
        
        plt.show()

mainFunc()
