import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt



class mainFunc():
    def __init__(self):
        self.proc_LoadData()
        self.proc_TreatData()
        
        self.proc_TrainModel()
    
        self.proc_ShowDataGraph()
        
        self.subproc_ElbolModel()
        
        
    
    def proc_LoadData(self, test=False):
        self.dataBin = pd.read_csv('csv/income.csv')
        if(test): print(self.dataBin)
        
        
        
    
    def proc_TreatData(self, test=False):
        scaler = MinMaxScaler()
        scaler.fit(self.dataBin[['Age']])
        
        self.dataBin['Age'] = scaler.transform(self.dataBin[['Age']])
        
        scaler.fit(self.dataBin[['Income($)']])
        
        self.dataBin['Income($)'] = scaler.transform(self.dataBin[['Income($)']])
        
        
        
    
    def proc_TrainModel(self, randomize = False):
        
        self.model = KMeans(n_clusters=3)
        self.model.fit(self.dataBin[['Age']], self.dataBin['Income($)'])
        self.dataBin['category'] = self.model.fit_predict(self.dataBin[['Age', 'Income($)']])
        
        print(self.dataBin)
    
    
    
    def proc_ShowDataGraph(self):
        g1 = self.dataBin[self.dataBin['category'] == 0]
        g2 = self.dataBin[self.dataBin['category'] == 1]
        g3 = self.dataBin[self.dataBin['category'] == 2]
        
        plt.scatter(g1[['Age']], g1['Income($)'], color='red')
        plt.scatter(g2[['Age']], g2['Income($)'], color='green')
        plt.scatter(g3[['Age']], g3['Income($)'], color='blue')
        
        plt.scatter(self.model.cluster_centers_[:,0],self.model.cluster_centers_[:,1], color='purple', marker='*')
        plt.xlabel('Age')
        plt.ylabel('Income($)')
        
        plt.show()
        
    def subproc_ElbolModel(self):
        sse = list()
        
        k_rng = range(1,10+1)
        
        for i in k_rng:
            model = KMeans(n_clusters=i)
            model.fit(self.dataBin[['Age']],self.dataBin['Income($)'])
            sse.append(model.inertia_)
            
        print(sse)
        
        plt.plot(k_rng, sse)
        plt.xlabel('K')
        plt.ylabel('Sum of Squared Errors')
        
        plt.show()
            
    
    
mainFunc()