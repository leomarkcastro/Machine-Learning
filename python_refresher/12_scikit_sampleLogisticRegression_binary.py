import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

class mainFunc():
    def __init__(self):
        self.proc_LoadData()
        self.proc_TreatData()
        self.proc_TrainModel()
        self.proc_GraphModel()
    
    def proc_LoadData(self):
        self.dataBin = pd.read_csv("csv/insurance_data.csv")
        print(self.dataBin)
    
    def proc_TreatData(self):
        X = self.dataBin[['age']]
        y = self.dataBin['bought_insurance']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, test_size = 0.2)
    
    def proc_TrainModel(self):
        self.model = LogisticRegression(max_iter=100)
        self.model.fit(self.X_train, self.y_train)
        
        print("Accuracy: ", self.model.score(self.X_test, self.y_test))
        print("coef:", self.model.coef_)
        print("intercept: ", self.model.intercept_)
    
    def proc_GraphModel(self):
        plt.scatter(self.dataBin['age'], self.dataBin['bought_insurance'], marker='x', color='red')
        #plt.plot(self.dataBin['age'], self.model.predict(self.dataBin[['age']]),color='blue')
        
        plt.show()
    
    
mainFunc()