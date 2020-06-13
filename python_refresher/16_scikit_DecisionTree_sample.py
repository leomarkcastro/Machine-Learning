import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

class mainFunc():
    def __init__(self):
        self.proc_LoadData()
        self.proc_ProcessData()
        self.proc_TrainModel()
        
    def proc_LoadData(self):
        self.dataBin = pd.read_csv("csv/salaries.csv")
        
    def proc_ProcessData(self):
        le_comp = LabelEncoder()
        le_job = LabelEncoder()
        le_deg = LabelEncoder()
        
        self.dataBin['company_n'] = le_comp.fit_transform(self.dataBin['company'])
        self.dataBin['job_n'] = le_comp.fit_transform(self.dataBin['job'])
        self.dataBin['degree_n'] = le_comp.fit_transform(self.dataBin['degree'])
        
        self.dataBin = self.dataBin.drop(['company','job','degree'],axis='columns')
    
    
    def proc_TrainModel(self):
        y = self.dataBin['salary_more_then_100k']
        X = self.dataBin.drop(['salary_more_then_100k'], axis='columns')
        
        self.model = tree.DecisionTreeClassifier()
        self.model.fit(X, y)
        
        print(self.model.score(X,y))
    
    
    
mainFunc()