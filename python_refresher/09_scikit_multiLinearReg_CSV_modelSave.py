import pandas as pd
import math
import matplotlib.pyplot as plt
from word2number import w2n
from sklearn import linear_model
import joblib

class mainFunc():
    def __init__(self):
        self.proc_LoadModel()
        
        #self.proc_LoadData()
        #self.proc_TreatData()
        
        #self.proc_LinearTraining()
        
        #self.proc_DisplayData()         #multivariate regressions are kind of hard to graph
        
        self.proc_TestData()
        
        #self.proc_SaveModel()
        
    def proc_LoadData(self):
        self.dataBin = pd.read_csv("csv/hiring.csv")
        print(self.dataBin)
        
    def proc_TreatData(self):
        self.dataBin['experience'] = self.dataBin['experience'].fillna('zero')
        new_list = pd.array(self.dataBin['experience'])
        for i in range(len(self.dataBin['experience'])):
           new_list[i] = w2n.word_to_num(self.dataBin['experience'][i])
        self.dataBin['experience'] = new_list
            
        testScore = math.floor(self.dataBin['test_score(out of 10)'].median())
        self.dataBin['test_score(out of 10)'] = self.dataBin['test_score(out of 10)'].fillna(testScore)
        print()
        print(self.dataBin)
        
    def proc_LinearTraining(self):
        self.reg = linear_model.LinearRegression()
        self.reg.fit(self.dataBin[['experience','test_score(out of 10)', 'interview_score(out of 10)']], self.dataBin['salary($)'])
        
        print()
        print('coef: ', self.reg.coef_)
        print('intercep: ', self.reg.intercept_)
    
    def proc_DisplayData(self):

        plt.xlabel("year")
        plt.ylabel("per capita income (US$)")
        plt.show()
        
    def proc_TestData(self):
        
        print()
        
        vx = 2
        vy = 9
        vz = 6
        
        print("exp: ", vx, "score: ", vy, "interview: ", vz)
        print("salary: ", self.reg.predict([[vx,vy,vz]]))
        
        print()
        
        vx = 12
        vy = 10
        vz = 10
        
        print("exp: ", vx, "score: ", vy, "interview: ", vz)
        print("salary: ", self.reg.predict([[vx,vy,vz]]))
        
    def proc_SaveModel(self):
        joblib.dump(self.reg, "models/trainedModel_01.joblib")
    
    def proc_LoadModel(self):
        self.reg = joblib.load("models/trainedModel_01.joblib")
    
mainFunc()