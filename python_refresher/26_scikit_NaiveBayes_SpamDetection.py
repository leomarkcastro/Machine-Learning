import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

class mainFunc():
    def __init__(self):
        self.proc_LoadData()
        
        #self.proc_TreatData()
        
        
        self.proc_TrainModel()
    
        
    
    def proc_LoadData(self, test=False):
        self.dataBin = pd.read_csv('csv/spam.csv')
        
        if(test): 
            print(self.dataBin)
            print()
            print(self.dataBin.groupby('Category').describe())
            
            
        self.dataBin['Spam'] = self.dataBin['Category'].apply(lambda x: 1 if x == 'spam' else 0) 
        
    
    def proc_TreatData(self, test=False):
        pass
        
    def proc_TrainModel(self, randomize = False):
        
        X_a, X_b, y_a, y_b = train_test_split(self.dataBin.Message, self.dataBin.Spam, test_size =0.25  )
        
        vectorizer = CountVectorizer()
        X_train_cv = vectorizer.fit_transform(X_a.values)
        
        self.model = MultinomialNB()
        self.model.fit(X_train_cv, y_a)
        
        emails = [
            'Hey Sir! I got a very generous discount for you, just visit our website and claim your free 20% discount',
            "Up to 20% discount, exclusive offer just for you! Don't miss the reward"
            ]
        
        email_c = vectorizer.transform(emails)
        
        print(self.model.predict(email_c))
        
        
    
    
mainFunc()