import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

from sklearn.datasets import load_iris

class mainFunc():
    def __init__(self):
        self.proc_LoadData()
        
        self.proc_TreatData()
        
        self.proc_GraphPreData()
        
        self.proc_TrainModel()
    
        self.proc_ShowDataGraph()
        
        self.subproc_ElbolModel()
        
    
    def proc_LoadData(self, test=False):
        data = load_iris()
        

        self.dataBin = pd.DataFrame(data.data, columns=data.feature_names)
        
        self.dataBin['target'] = data.target
        
        self.dataBin = self.dataBin.drop(['sepal length (cm)','sepal width (cm)'], axis='columns')
        
        if(test): print(self.dataBin)
        
    
    def proc_TreatData(self, test=False):
        scaler = MinMaxScaler()
        scaler.fit(self.dataBin[['petal length (cm)']])
        
        self.dataBin['petal length (cm)'] = scaler.transform(self.dataBin[['petal length (cm)']])
        
        scaler.fit(self.dataBin[['petal width (cm)']])
        
        self.dataBin['petal width (cm)'] = scaler.transform(self.dataBin[['petal width (cm)']])
        
        
    def proc_GraphPreData(self):
    
        plt.scatter(self.dataBin[['petal length (cm)']], self.dataBin['petal width (cm)'])
        plt.show()
        
    
    def proc_TrainModel(self, randomize = False):
        
        self.model = KMeans(n_clusters=3)
        self.model.fit(self.dataBin[['petal length (cm)']], self.dataBin['petal width (cm)'])
        self.dataBin['category'] = self.model.fit_predict(self.dataBin[['petal length (cm)', 'petal width (cm)']])
        
        print(self.dataBin)
    
    
    
    def proc_ShowDataGraph(self):
        g1 = self.dataBin[self.dataBin['target'] == 0]
        g2 = self.dataBin[self.dataBin['target'] == 1]
        g3 = self.dataBin[self.dataBin['target'] == 2]
        
        f = plt.figure(0)
        plt.scatter(g1[['petal length (cm)']], g1['petal width (cm)'], color='red')
        plt.scatter(g2[['petal length (cm)']], g2['petal width (cm)'], color='green')
        plt.scatter(g3[['petal length (cm)']], g3['petal width (cm)'], color='blue')
        
        plt.xlabel('petal length (cm)')
        plt.ylabel('petal width (cm)')
        
        
        
        g1 = self.dataBin[self.dataBin['category'] == 0]
        g2 = self.dataBin[self.dataBin['category'] == 1]
        g3 = self.dataBin[self.dataBin['category'] == 2]
        
        g = plt.figure(1)
        plt.scatter(g1[['petal length (cm)']], g1['petal width (cm)'], color='red')
        plt.scatter(g2[['petal length (cm)']], g2['petal width (cm)'], color='green')
        plt.scatter(g3[['petal length (cm)']], g3['petal width (cm)'], color='blue')
        
        
        plt.scatter(self.model.cluster_centers_[:,0],self.model.cluster_centers_[:,1], color='purple', marker='*')
        
        f.show()
        g.show()
        
        plt.show()
        
    def subproc_ElbolModel(self):
        sse = list()
        
        k_rng = range(1,10+1)
        
        for i in k_rng:
            model = KMeans(n_clusters=i)
            model.fit(self.dataBin[['petal length (cm)']],self.dataBin['petal width (cm)'])
            sse.append(model.inertia_)
            
        print(sse)
        
        plt.plot(k_rng, sse)
        plt.xlabel('petal length (cm)')
        plt.ylabel('petal width (cm)')
        
        plt.show()
            
    
    
mainFunc()