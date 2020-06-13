import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.datasets import load_wine


class mainFunc():
    def __init__(self):
        self.proc_LoadData()
        
        #self.proc_TreatData()
        
        self.proc_TrainModel()
    
        
    
    def proc_LoadData(self, test=True):
        data = load_wine()
        
        self.dataBin = pd.DataFrame(data.data, columns = data.feature_names)
        self.dataBin['target'] = data.target
        
        if(test): 
            print(self.dataBin)
            print()
            
            
    
    def proc_TreatData(self, test=False):
        pass
        
    def proc_TrainModel(self, randomize = False):
        
        self.X = self.dataBin.drop('target', axis='columns')
        self.y= self.dataBin.target
        
        X_a, X_b, y_a, y_b = train_test_split(self.X, self.y, test_size =0.25  )
        
        self.model = GaussianNB()
        self.model.fit(X_a, y_a)
        
        print(self.model.score(X_b,y_b))

        self.modelb = MultinomialNB()
        self.modelb.fit(X_a, y_a)
        
        print(self.modelb.score(X_b,y_b))

        
        
    
    
mainFunc()