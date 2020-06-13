import numpy as np


class mainFunc:
    def __init__(self):
        self.mainProcess()
        
        
    def step_DecVars(self):
        self.a_weight = np.round(np.random.normal(70,10,100),2)
        self.a_height = np.round(np.random.normal(1.75,0.2,100),2)
        
        #self.step_DecVars_print()
        
    def step_DecVars_print(self):
        print("=======Weight=======")
        print(self.a_weight)
        print("Min: " + str(min(self.a_weight)))
        print("Max: " + str(max(self.a_weight)))
        
        print("\n\n=======Height=======")
        print(self.a_height)
        print("Min: " + str(min(self.a_height)))
        print("Max: " + str(max(self.a_height)))
    
    def step_ProcVars(self):
        pass
    
    def mainProcess(self):
        
        self.step_DecVars()
        self.step_ProcVars()
        
        
mainFunc()