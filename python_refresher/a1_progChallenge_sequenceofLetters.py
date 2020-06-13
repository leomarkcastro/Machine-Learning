
class mainFunc():
    def __init__(self):
        self.mainProcess()
        
    def mainProcess(self):
        ctrState = 0
        incre = 1
        
        for i in range(100):
            resPrt = ''
            
            x = ctrState
            
            while (x // 26 > 0):
                resPrt += chr(ord('A') + (x % 26))
                x //= 26
            else:
                if (len(resPrt) == 0):
                    resPrt += chr(ord('A') + (x % 26))
                else:
                    resPrt += chr(ord('A') + (x % 26))
                
                
            print("CtrState: ", ctrState, " Generated: ",resPrt[::-1])
            
            ctrState += incre
            incre += 1
    
    
mainFunc()