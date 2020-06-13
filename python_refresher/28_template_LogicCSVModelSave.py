import pandas as pd
import joblib

from sklearn.model_selection import KFold, train_test_split

from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree._classes import DecisionTreeClassifier


class mainFunc:
    def __init__(self):
        #self.proc_LoadModel()
        
        self.proc_MainMachineLearningLogic()
        
        #self.proc_CSV_Export()
        
        #self.proc_SaveModel()
    
    def met_MachineModelScore(self, model, X_train, y_train, X_test, y_test):
            model.fit(X_train, y_train)
            return model.score(X_test, y_test)
        
    def met_MachineModel(self, model, X_train, y_train):
        x = model.fit(X_train, y_train)
        return x
        
    def met_kFolding_MachineTest(self, X_set, y_set, model = [SVC()], k_splits = 4, shuf = False ,ret_val = 'Scores'):
            
        folder = KFold(n_splits=k_splits, shuffle=shuf)
    
        model_Score = dict()
        model_Machine = dict()
    
        for mod in model:
            model_Score[mod.__class__.__name__] = []
            model_Machine[mod.__class__.__name__] = mod
        
        for train_i, test_i in folder.split(X_set, y_set):
            X_train, X_test, y_train, y_test = X_set[train_i], X_set[test_i], y_set[train_i], y_set[test_i] 
            
            print("Train Size: ", len(y_train), " Test Size:",len(y_test))
            
            for mod in model:
                
                model_Machine[mod.__class__.__name__] = (self.met_MachineModel(model_Machine[mod.__class__.__name__], X_train, y_train))
                model_Score[mod.__class__.__name__].append(self.met_MachineModelScore(model_Machine[mod.__class__.__name__], X_train, y_train, X_test, y_test))
        
        print()
        
        if (ret_val == 'Scores'):      
            return model_Score
        elif (ret_val == 'Models'):
            return model_Machine
        elif (ret_val == 'All'):
            return model_Score, model_Machine
    
    def met_trainTest_MachineTest(self, X_set, y_set, model = [SVC], test_size = 0.2, shuf = False, ret_val = 'Scores'):  
            
        model_Score = dict()
        model_Machine = dict()
    
        for mod in model:
            model_Score[mod.__class__.__name__] = []
            model_Machine[mod.__class__.__name__] = mod
        
        X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, test_size=test_size, shuffle=shuf)
            
        print("Train Size: ", len(y_train), " Test Size:",len(y_test))
        print()
        
        for mod in model:
            
            model_Machine[mod.__class__.__name__] = (self.met_MachineModel(model_Machine[mod.__class__.__name__], X_train, y_train))
            model_Score[mod.__class__.__name__].append(self.met_MachineModelScore(model_Machine[mod.__class__.__name__], X_train, y_train, X_test, y_test))
            
        
        if (ret_val == 'Scores'):      
            return model_Score
        elif (ret_val == 'Models'):
            return model_Machine
        elif (ret_val == 'All'):
            return model_Score, model_Machine
           
    def met_basic_MachineTest(self, X_set, y_set, model = [SVC], ret_val = 'Scores'):  
            
        model_Score = dict()
        model_Machine = dict()
    
        for mod in model:
            model_Score[mod.__class__.__name__] = []
            model_Machine[mod.__class__.__name__] = mod
            
        for mod in model:
            
            model_Machine[mod.__class__.__name__] = (self.met_MachineModel(model_Machine[mod.__class__.__name__], X_set, y_set))
            model_Score[mod.__class__.__name__].append(self.met_MachineModelScore(model_Machine[mod.__class__.__name__], X_set, y_set, X_set, y_set))
            
        
        if (ret_val == 'Scores'):      
            return model_Score
        elif (ret_val == 'Models'):
            return model_Machine
        elif (ret_val == 'All'):
            return model_Score, model_Machine
    
    
    def proc_MainMachineLearningLogic(self):

        def sampleTesting():
            
            print('\n=====================================================\n')
            print ('k Folding\n')
            
            data = load_iris()
            
            mod_Score, mod_Data = self.met_kFolding_MachineTest(data.data, data.target, [LogisticRegression(max_iter=1000), DecisionTreeClassifier(), SVC(max_iter=1000)], k_splits=5, ret_val = 'All')
            
            
            for key in mod_Score:
                print(key)
                print(mod_Score[key])
            
            print()
            
            for keys in mod_Data:
                print(keys," : ", mod_Data[keys].score(data.data, data.target))
                
                
            print('\n=====================================================\n')
            print ('Train Test Folding\n')
            
            data = load_iris()
            
            mod_Score, mod_Data = self.met_trainTest_MachineTest(data.data, data.target, [LogisticRegression(max_iter=1000), DecisionTreeClassifier(), SVC(max_iter=1000)], test_size=0.2, ret_val = 'All')
            
            
            for key in mod_Score:
                print(key)
                print(mod_Score[key])
            
            print()
            
            for keys in mod_Data:
                print(keys," : ", mod_Data[keys].score(data.data, data.target))
                
                
            print('\n=====================================================\n')
            print ('Basic Machine Learning\n')
            
            data = load_iris()
            
            mod_Score, mod_Data = self.met_basic_MachineTest(data.data, data.target, [LogisticRegression(max_iter=1000), DecisionTreeClassifier(), SVC(max_iter=1000)], ret_val = 'All')
            
            for key in mod_Score:
                print(key)
                print(mod_Score[key])
            
            print()
            
            for keys in mod_Data:
                print(keys," : ", mod_Data[keys].score(data.data, data.target))
        
        def sampleTesting2():
            
            preData = load_iris()
            
            self.dataBin = pd.DataFrame(preData.data, columns = preData.feature_names)
            #print(self.dataBin)
            
            self.model = self.met_kFolding_MachineTest(preData.data, preData.target, [SVC(max_iter=1000, C=10, gamma=0.1)], k_splits = 4, shuf=True, ret_val = 'Models')
            
            self.dataBin['target'] = preData.target
            self.dataBin['guess'] = self.model['SVC'].predict(preData.data)
            self.dataBin['target_names'] = self.dataBin.target.apply(lambda x: preData.target_names[x])
            self.dataBin['guess_names'] = self.dataBin.guess.apply(lambda x: preData.target_names[x])
            
            self.dataBin = self.dataBin.drop(['target', 'guess'], axis='columns')
            
            print(self.dataBin.head(10))
            print(self.model['SVC'].score(preData.data, preData.target))
            
            #self.proc_CSV_Export(self.dataBin, "csv/SampleExport.csv")
    
        sampleTesting()
    
    def proc_CSV_Export(self, to_sav, name, index=False):
        to_sav.to_csv(name, index=index)    
        
    def proc_SaveModel(self, to_sav, name):
        joblib.dump(to_sav, "models/" + name + ".joblib")
    
    def proc_LoadModel(self, name):
        self.model = joblib.load("models/" + name + ".joblib")
    
mainFunc()