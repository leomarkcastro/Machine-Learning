import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd

class mainFunc:
    def __init__(self):
        self.mainProcess()
    
    def mainProcess(self):
        self.proc_DecVars()
        self.proc_PrintVars()
        self.proc_ProcVars()
        self.proc_GraphVars()
        
    def proc_DecVars(self):
        self.dataBin = pd.read_csv("csv/1000 Sales Records.csv")
        
    def proc_PrintVars(self):
        print(self.dataBin.keys())
        
    def proc_ProcVars(self):
        pass
    
    def proc_GraphVars(self):
        sns.swarmplot(x = "Region", y = "Units Sold", data = self.dataBin)
        plt.show()
        
            
mainFunc()