import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn

class mainFunc():
    def __init__(self):
        self.proc_LoadData()
        
        #self.test_LoadData()
        
        self.proc_TreatData()
        
        self.proc_TrainModel()
        
        self.proc_EvaluateModel()
    
    def proc_LoadData(self):
        self.dataBin = load_iris()
        
    
    def test_LoadData(self):
        print(dir(self.dataBin))
        
        x = self.dataBin
        
        print(x.feature_names)
        print(x.data[0])
        
        print(x.target_names)
        print(x.target[0])
        

    def proc_TreatData(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataBin.data, self.dataBin.target, test_size = 0.2)

    
    def proc_TrainModel(self):
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)
        
        print("Score: ", self.model.score(self.X_test, self.y_test))
    
    def proc_EvaluateModel(self):
        
        y_predicted = self.model.predict(self.X_test)
        
        cm = confusion_matrix(self.y_test, y_predicted)
        
        plt.figure(figsize=(10,7))
        sn.heatmap(cm,annot=True)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        
        plt.show()
    
mainFunc()