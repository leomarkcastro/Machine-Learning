import pandas as pd
from sklearn.datasets import load_digits

from sklearn.model_selection import KFold, train_test_split

from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


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
        
        def get_score(model, X_train, X_test, y_train, y_test):
            model.fit(X_train, y_train)
            return model.score(X_test, y_test)
        
        folder = KFold(n_splits=3)
        
        score_LogReg = list()
        score_SVC = list()
        score_RFC = list()
        
        lData = self.data
        
        for train_i, test_i in folder.split(lData.data):
            X_train, X_test, y_train, y_test = lData.data[train_i], lData.data[test_i], lData.target[train_i], lData.target[test_i] 

            score_LogReg.append(get_score(LogisticRegression(), X_train, X_test, y_train, y_test))
            score_SVC.append(get_score(SVC(), X_train, X_test, y_train, y_test))
            score_RFC.append(get_score(RandomForestClassifier(), X_train, X_test, y_train, y_test))
    
        print("LogReg: ", score_LogReg)
        print("SVC: ", score_SVC)
        print("RFC: ", score_RFC)
    
    def test_Data(self):
        print(self.dataBin)
    
mainFunc()