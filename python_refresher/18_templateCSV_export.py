import pandas as pd


class mainFunc:
    def __init__(self):
        self.proc_DummyLogic()
        self.proc_CSV_Export()
    
    def proc_DummyLogic(self):
        x1 = list()
        x2 = list()
        
        self.dataBin = pd.DataFrame()
        
        for i in range(0,101):
            x1.append(i)
            x2.append(i+2)
            
        self.dataBin['id'] = x1
        self.dataBin['new_number'] = x2
        
        print(self.dataBin)
    
    def proc_CSV_Export(self):
        self.dataBin.to_csv('DummySubmission_100620_LMdcC.csv', index=False)    
    
mainFunc()