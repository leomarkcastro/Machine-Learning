from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_iris
import pandas as pd
import joblib
from sklearn.svm._classes import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree._classes import DecisionTreeClassifier

class mainFunc():
    def __init__(self):
        self.proc_LoadData()
        self.proc_TreatData()
        self.proc_CreateModel()
        self.proc_GeneratePredictions()
    
    
    
    
    def proc_LoadData(self):
        self.dataBin = load_iris()
    
    def proc_TreatData(self):
        pass
    
    def proc_CreateModel(self):
        
        data = self.dataBin
        
        def proc_Sample1():
            if (False): print(DecisionTreeClassifier().get_params())
            
            if (True): self.proc_BruteForceModel_Random(data.data, data.target, SVC(max_iter=1000), {
                'C': [0.01, 0.1, 1, 10, 100],
                'kernel' : ['linear', 'rbf'],
                }, cases=5, iterations=6, return_type='print')
            
        def proc_Sample2():
            self.model = self.proc_BruteForceModel(data.data, data.target, SVC(max_iter=1000), {}, cases=7, return_type='model')
            
            print(self.model.score(data.data, data.target))
         
        
        proc_Sample1()
        
        
    def proc_GeneratePredictions(self):
        pass
    
    
    
    print(end = '')
    
    
    ##################################################################################################
    ##################################################################################################
    # Don't Touch Anything Beyond this Line! You might get tangled up with the spaghetti
    ##################################################################################################
    ##################################################################################################
        
        
        
    #Brute Force Model either 'print', 'export', 'model'
        
    def proc_BruteForceModel(self, Xtest, ytest, modelToTest, parameters, cases = 5, return_type = 'print'):
        
        clf = GridSearchCV(modelToTest, parameters, cv=cases, return_train_score = False)
        
        clf.fit(Xtest, ytest)
        
        if (return_type == 'print'):
            to_sav= pd.DataFrame(clf.cv_results_)[['params', 'mean_test_score', 'rank_test_score']]
            print(to_sav)
            print()
            print('Best Score :', clf.best_score_)
            print('Best Param :', clf.best_params_)
        elif (return_type == 'export'):
            to_sav= pd.DataFrame(clf.cv_results_)[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
            self.proc_CSV_Export(to_sav, "HyperParameter_"+modelToTest.__class__.__name__+'.csv', index=True)
        elif (return_type == 'model'):
            print('---------------------------------')
            print('Best Score :', clf.best_score_)
            print('Best Param :', clf.best_params_)
            return clf.best_estimator_
             
    def proc_BruteForceModel_Random(self, Xtest, ytest, modelToTest, parameters, cases = 5, iterations = 10,return_type = 'print'):
        
        clf = RandomizedSearchCV(modelToTest, parameters, cv=cases, return_train_score = False, n_iter=iterations)
        
        clf.fit(Xtest, ytest)
        
        if (return_type == 'print'):
            to_sav= pd.DataFrame(clf.cv_results_)[['params', 'mean_test_score', 'rank_test_score']]
            print(to_sav)
            print()
            print('Best Score :', clf.best_score_)
            print('Best Param :', clf.best_params_)
        elif (return_type == 'export'):
            to_sav= pd.DataFrame(clf.cv_results_)[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
            self.proc_CSV_Export(to_sav, "HyperParameter_"+modelToTest.__class__.__name__+'.csv', index=True)
        elif (return_type == 'model'):
            print('---------------------------------')
            print('Best Score :', clf.best_score_)
            print('Best Param :', clf.best_params_)
            return clf.best_estimator_    
    
    
    
    def proc_CSV_Export(self, to_sav, name, index=False):
        to_sav.to_csv('csv/'+name, index=index)    
        
    def proc_SaveModel(self, to_sav, name):
        joblib.dump(to_sav, "models/" + name + ".joblib")
    
    def proc_LoadModel(self, name):
        self.model = joblib.load("models/" + name + ".joblib")
        
mainFunc()
