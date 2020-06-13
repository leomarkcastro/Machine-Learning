import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class mainFunc:
    def __init__(self):
        self.mainProcess()
    
    def mainProcess(self):
        self.proc_DecVars()
        self.proc_DispVars()
        self.proc_ProcVars()
        self.proc_GraphVars()
        
    def proc_DecVars(self):
        self.grades = np.round(np.random.normal(50,5,100), 2)
        self.happiness = np.round(np.random.normal(50,20,100), 2)
        
    def proc_DispVars(self):
        print ("Grades")
        print(self.grades)
        print ("Min: ", min(self.grades))
        print ("Max: ", max(self.grades))
        
        print ("\n\nHappiness")
        print(self.happiness)
        print ("Min: ", min(self.happiness))
        print ("Max: ", max(self.happiness))
    
    def proc_ProcVars(self):
        pass
    
    def proc_GraphVars(self):
        sns.set()
        
        plt.hist(self.grades, bins=50)
        plt.show()
    
    def proc_GraphVars2(self):
        x = np.sort(self.grades)
        y = np.arange(1, len(x)+1) / len(x)
        
        plt.plot(x, y, marker = '.', linestyle = 'none')
        
        x = np.sort(self.happiness)
        y = np.arange(1, len(x)+1) / len(x)
        
        plt.plot(x, y, marker = '.', linestyle = 'none')

        
        plt.xlabel('grade')
        plt.ylabel('ECDF')
        plt.margins(0.02)
        
        
        plt.show()
        
            
mainFunc()