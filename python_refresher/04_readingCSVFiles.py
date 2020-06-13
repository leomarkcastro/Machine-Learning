import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def spaceDown(size = 1):
    for i in range(size):
        print()

class mainFunc:
    def __init__(self):
        self.mainProcess()
    
    def mainProcess(self):
        self.proc_DecVars()
        self.proc_DispVars()
        self.proc_ProcVars()
        self.proc_GraphVars()
        
    def proc_DecVars(self):
        self.dataBin = pd.read_csv("csv/1000 Sales Records.csv")
        
    def proc_DispVars(self):
        #print(self.dataBin.keys())
        pass
    
    def proc_ProcVars(self):
        
        def frequency(category):
            freqCount = dict()
            
            for item in self.dataBin[category]:
                if item in freqCount:
                    freqCount[item] += 1
                else:
                    freqCount[item] = 1
                    
            return freqCount
        
        spaceDown()
        freqCount = frequency("Sales Channel")
        print(freqCount)
        
        spaceDown()
        freqCount = frequency("Order Priority")
        print(freqCount)
        
        spaceDown()
        freqCount = frequency("Region")
        print(freqCount)
        
        
    
    def proc_GraphVars(self):
        sns.swarmplot(x = 'Sales Channel', y = 'Order Priority', data=self.dataBin)
        plt.show()
        
            
mainFunc()