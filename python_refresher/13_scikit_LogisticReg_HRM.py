import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

class mainFunc():
    def __init__(self):
        self.proc_LoadData()
        self.proc_AnalyseData()
        self.proc_TreatData()
        self.proc_TrainModel()
        self.proc_GraphModel()
    
    def proc_LoadData(self):
        self.dataBin = pd.read_csv("csv/HR_comma_sep.csv")
        
    def proc_AnalyseData(self):
        print("Left: ", self.dataBin[self.dataBin.left==1].shape)
        
        print("Retention: ", self.dataBin[self.dataBin.left==0].shape)
        
        print()
        
        print(self.dataBin.groupby(self.dataBin.left).mean())
    
    def proc_TreatData(self):
        self.dataBinA = self.dataBin[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
        sal_dummies = pd.get_dummies(self.dataBinA.salary, prefix = 'salary')
        self.dataBinA = pd.concat([self.dataBinA, sal_dummies], axis='columns')
        self.dataBinA = self.dataBinA.drop(['salary', 'salary_high'], axis='columns')
        
        self.X = self.dataBinA
        self.y = self.dataBin.left
    
    def proc_TrainModel(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size = 0.2)
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)
        
        print(self.model.score(X_test, y_test))
    
    def proc_GraphModel(self):
        pd.crosstab(self.dataBin.salary, self.dataBin.left).plot(kind='bar')
        pd.crosstab(self.dataBin.Department, self.dataBin.left).plot(kind='bar')
        
        plt.show()
    
    
mainFunc()