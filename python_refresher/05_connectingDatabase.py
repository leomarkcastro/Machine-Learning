import sqlalchemy

class mainClass:
    def __init__(self):
        self.mainProcess()
    
    
    def mainProcess(self):
        self.proc_SetUpConnection(user = 'root', password = 'root', database = 'cpe2b')
        self.proc_Query("select * from cpe2b.stud_info", "print")
        

    #Set Up the connection to MySql database server 
    def proc_SetUpConnection(self, user = "root", password = "root", link = "localhost:3306", database = "panels"):
        ## create_engine format: "mysql+pymysql://user:password@link/database"
        try:
            self.engine = sqlalchemy.create_engine('mysql+pymysql://' + user + ':' + password + '@' + link + '/' + database)
            self.connection = self.engine.connect()
        except RuntimeError:
            print("\nError while connecting to database server. Please check your parameters\n")
        
    #Query input with option to print result or take a return value
    def proc_Query(self, query = "select * from panels.grades", resType = "null"):
        
        
        try:
            result_proxy = self.connection.execute(query)
            
            if (resType != "null"): 
                results = result_proxy.fetchall()
            
            if (resType == "print"):
                print(results[0].keys(),end="\n\n")
                [print(item) for item in results]
                
            elif (resType == "keys"):
                [print(item) for item in results[0].keys()]
                    
            elif (resType == "return"):
                return results
            
            else:
                pass
            
        except AttributeError:
            print("\nError fetching data due to connection error\n")
    

    
#mainClass()

metadata = sqlalchemy.MetaData()
census = sqlalchemy.Table('census', metadata, autoload=True, autol)

