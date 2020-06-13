import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

class mainFunc():
    def __init__(self):
        self.proc_LoadData()
        self.proc_ProcessData()
        self.proc_TrainModel()
        
    def proc_LoadData(self):
        self.dataBin = pd.read_csv('csv/titanic.csv')
        
    def proc_ProcessData(self):
        self.dataBin = self.dataBin[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
        
        age_median = self.dataBin['Age'].median()
        
        self.dataBin['Age'] = self.dataBin['Age'].fillna(age_median)
        
        le_sex = LabelEncoder()
        
        self.dataBin['Sex_n'] = le_sex.fit_transform(self.dataBin['Sex'])

        self.dataBin = self.dataBin.drop('Sex', axis='columns')
        
        print(self.dataBin.head(10))
    
    def proc_TrainModel(self):
        y = self.dataBin['Survived']
        X = self.dataBin.drop(['Survived'], axis='columns')
        
        self.model = tree.DecisionTreeClassifier()
        self.model.fit(X, y)
        
        print(self.model.score(X,y))
        
        trial_value = [1,35,70.25,0]
        
        print('\nPclass: ', trial_value[0], '\nAge: ', trial_value[1], '\nFare: ', trial_value[2], '\nSex_n: ', trial_value[3], "\nSurvived(prediction): ", self.model.predict([trial_value]))
    
    
    
mainFunc()