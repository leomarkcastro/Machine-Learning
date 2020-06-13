import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

class mainFunc():
    def __init__(self):
        self.proc_LoadData()
        
        self.proc_TreatData()
        
        
        self.proc_TrainModel()
    
        
    
    def proc_LoadData(self, test=False):
        self.dataBin = pd.read_csv('csv/titanic.csv')
        
        self.dataBin = self.dataBin.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns')
        
        if(test): print(self.dataBin)
        
    
    def proc_TreatData(self, test=False):
        dummies = pd.get_dummies(self.dataBin['Sex'])
        
        self.dataBin = pd.concat([self.dataBin, dummies], axis='columns')
        
        self.dataBin = self.dataBin.drop(['Sex'], axis='columns')
        
        self.dataBin['Age'] = self.dataBin['Age'].fillna(self.dataBin['Age'].mean())
        
        print(self.dataBin.head(20))
        
        
    def proc_TrainModel(self, randomize = False):
        
        self.X = self.dataBin.drop(['Survived'], axis='columns')
        self.y = self.dataBin['Survived']
        
        X_a, X_b, y_a, y_b = train_test_split(self.X, self.y, test_size = 0.2)
        
        self.model = GaussianNB()
        
        self.model.fit(X_a, y_a)
        
        print(self.model.score(X_b, y_b))
    
        print(self.model.predict_proba(self.X[:10]))
    
    
mainFunc()