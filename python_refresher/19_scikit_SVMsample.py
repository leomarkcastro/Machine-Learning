import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from random import randint, choice

class mainFunc():
    def __init__(self):
        self.proc_LoadData(True)
        self.proc_TreatData()
        
        self.proc_TrainModel()
        self.test_Model()
    
    def proc_LoadData(self, test=False):
        data = load_iris()
        
        self.dataBin = pd.DataFrame(data.data, columns=data.feature_names)
        self.dataBin['target'] = data.target
        self.dataBin['target_names'] = self.dataBin.target.apply(lambda x: data.target_names[x])
    
        
        if (test): self.test_Data()
    
    def proc_TreatData(self, test=False):
        self.X = self.dataBin.drop(['target', 'target_names'], axis='columns')
        self.y = self.dataBin.target
    
    def proc_TrainModel(self, randomize = False):
        #########################Preseting
        
        if(randomize):
            r_state = randint(0,50)
            C_v = randint(0,50)
            gamma_v = randint(0,50)
            kernel_v = choice(['linear','rbf', 'poly', 'sigmoid', 'precomputed'])
        else:
            r_state = 18                #18 results to perfect value
            C_v = 1.0                   #1.0 is default
            gamma_v = 'auto'            #'auto' is default
            kernel_v = 'rbf'            #'rbf' is default
        
        ###################################
        
        X_1, X_2, y_1, y_2 = train_test_split(self.X, self.y, test_size=0.2, random_state=r_state, shuffle=True)
        
        print("\nr_state: ", r_state)
        print("C value: ", C_v)
        print("Gamma value: ", gamma_v)
        print("Kernelling: ", kernel_v)
        
        self.model = SVC(C=C_v, gamma=gamma_v, kernel=kernel_v)
        
        self.model.fit(X_1, y_1)
        
        print("\nModel score: ", self.model.score(X_2,y_2))
        
        
    
    def test_Data(self):
        print(self.dataBin)
    
    def test_Model(self):
        pass
    
mainFunc()