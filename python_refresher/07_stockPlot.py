import numpy as np
import matplotlib.pyplot as plt

class mainFunc():
    def __init__(self):
        self.proc_DecVar()
        self.proc_GraphVar()
    
    def proc_DecVar(self):
        self.dataBin = list()
        
        self.dS = 5
        
        for setBox in range(self.dS):
            x = np.round(np.random.lognormal(5,(1/self.dS)*(setBox+1),100),2)
            if(setBox % 2 == 0): 
                #x = np.sort(x)
                pass
            print("Min: ", min(x)," Max: ", max(x))
            self.dataBin.append(x)
        
        self.totalSales = np.random.normal(0,0,100)
        
        for setBox in self.dataBin:
            self.totalSales += setBox
            print(self.totalSales)
                
        
    
    def proc_GraphVar(self):
        x = [item for item in range(0,100)]
        y = np.vstack([item for item in self.dataBin])
        sy = np.vstack(self.totalSales)

        labels = ["1 ", "2", "3"]
        
        plt.plot(x,sy)
        
        for i in range(self.dS):
            plt.plot(x,self.dataBin[i])
        
        fig, ax = plt.subplots()
        ax.stackplot(x, y)
        plt.show()
    
mainFunc()