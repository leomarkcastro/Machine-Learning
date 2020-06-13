import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


class mainFunc:
    def __init__(self):
        self.mainProcess()
    
    def mainProcess(self):
        self.proc_DecVars()
        self.proc_DispVars()
        self.proc_ProcVars()
        self.proc_GraphVars()
        
    def proc_DecVars(self):
        self.dataBin = dict()
        
        x = np.round(np.random.normal(50,50,100),2)
        self.dataBin['scores'] = x
        
        x = list()
        for i in range(100):
            x.append( random.choice(['a','b','c']) )
        self.dataBin['category'] = x
        
        
    def proc_DispVars(self):
        print(self.dataBin)
        
    def proc_ProcVars(self):
        pass
    
    def proc_GraphVars(self):
        sns.swarmplot(x = "category", y = "scores", data = self.dataBin)
        plt.show()
        
            
mainFunc()