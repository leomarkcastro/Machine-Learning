import pandas as pd
from sqlalchemy.sql.expression import false

class mainFunc:
    def __init__(self):
        self.proc_LoadData()
        self.proc_ExploreData()
        self.proc_CleanData()
        self.proc_FeatureFixingData()
    
    def proc_LoadData(self):
        self.dataBin = pd.read_csv('csv/datasetHousing.csv')
    
    def proc_ExploreData(self):
        data = self.dataBin
        print(data)
        print("\n=================================\n")
        
        #Count the total rows and columns
        print("Size and Columns\n\n", data.shape)
        print("\n=================================\n")
        
        #Count the instance of unique entries in the list
        print("Count of unique entries\n")
        print(data.groupby('area_type')['area_type'].agg('count'))
        print("\n=================================\n")
        
        #List all the unique entries of the list 
        print("List of unique entries\n")
        print(data['size'].unique())
        print("\n=================================\n")
        
        #Count the null in the database
        print("Count of null entries in all fields\n")
        print(data.isnull().sum())
        print("\n=================================\n")
        
    def proc_CleanData(self):
        #Extract all important fields
        self.dataBin = self.dataBin[['location', 'size', 'total_sqft', 'bath', 'price']]
        
        #Remove null entries
        self.dataBin = self.dataBin.dropna()
        print("\n=================================\n")
        
        #Convert size to bathroom
        self.dataBin['bhk'] = self.dataBin['size'].apply( lambda x: int( x.split()[0] ) )
        self.dataBin = self.dataBin.drop('size', axis = 'columns')
        
        def isFloat(x):
            try: float(x)
            except: return False
            return True
        
        def clean_getAve_sqft(x):
            tok = x.split('-')
            if (len(tok) == 2):
                return (float(tok[0]) + float(tok[1]))/2
            try:
                return float(x)
            except:
                return None
                
        #Look for rough data
        print(self.dataBin[~self.dataBin['total_sqft'].apply(lambda x: isFloat(x))].head(20))
        self.dataBin['total_sqft'] = self.dataBin['total_sqft'].apply(clean_getAve_sqft)
        print("\n=================================\n")
        
        #Check a profile
        print(self.dataBin.loc[30])
        print("\n=================================\n")
        
        #Remove null entries
        self.dataBin = self.dataBin.dropna()
        print(self.dataBin.isnull().sum())
        print("\n=================================\n")

    def proc_FeatureFixingData(self):
        
        #Cleaning and Standardizing the location
        self.dataBin.location = self.dataBin.location.apply(lambda x: x.strip())
        
        #Check which places are prevaluent. Those who are not will be categorized as 'other' places
        locList = (self.dataBin.groupby('location')['location'].agg('count').sort_values(ascending=False))
        print(locList)
        print("\n=================================\n")
        
        #Get all the entry with less than 10 occurence
        loc_lessthan10 = locList[locList <= 10]
        print(loc_lessthan10)
        print("\n=================================\n")
        
        #Replace all insignificant entries
        self.dataBin.location = self.dataBin.location.apply(lambda x: 'other' if x in loc_lessthan10 else x)
        
        print(self.dataBin.head(20))
    
mainFunc()  