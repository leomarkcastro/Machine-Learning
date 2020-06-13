import pandas as pd
from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier 


class mainFunc():
    def __init__(self):
        self.proc_LoadData()
        self.proc_TreatData()
        
        self.proc_TrainModel()
    
    def proc_LoadData(self, test=False):
        self.data = load_digits()
        
        self.dataBin = pd.DataFrame(self.data.data)
        self.dataBin['target'] = self.data.target
        
        if (test): self.test_Data()
    
    def proc_TreatData(self, test=False):
        self.X = self.dataBin.drop(['target'], axis='columns')
        self.y = self.dataBin.target
    
    def proc_TrainModel(self, randomize = False):
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
            
        self.model = RandomForestClassifier()
        
        self.model.fit(X_train, y_train)
        print(self.model.score(X_test, y_test))
        
        
    
    def test_Data(self):
        print(self.dataBin)
    
mainFunc()