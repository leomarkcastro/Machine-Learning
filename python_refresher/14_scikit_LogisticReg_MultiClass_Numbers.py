import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
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
        
        self.proc_GraphModel()
        
        
    
    def proc_LoadData(self):
        self.digits = load_digits()
    
    def test_LoadData(self):
        print(self.digits.data[0])

        plt.matshow(self.digits.images[0])
        plt.show()
    
    def proc_TreatData(self):
        self.X = self.digits.data
        self.y = self.digits.target
    
    def proc_TrainModel(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.1)
        
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)
        
        print(self.model.score(X_test, y_test))
        
        y_predicted = self.model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_predicted)
        
        plt.figure(figsize=(10,7))
        sn.heatmap(cm,annot=True)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        
        plt.show()
        
    
    def proc_GraphModel(self):
        pass
    
mainFunc()