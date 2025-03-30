import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataIngestion:
    
    def __init__(self):

        self.raw_data_path=os.path.join('artifacts','raw.csv')
        self.train_data_path=os.path.join('artifacts','train.csv')
        self.test_data_path=os.path.join('artifacts','test.csv')
    
    def data_ingestion(self):

        data=pd.read_csv(r'data\train.csv')

        os.makedirs(os.path.dirname(self.raw_data_path),exist_ok=True)
        data.to_csv(self.raw_data_path, index=False)

        train_data,test_data=train_test_split(data,test_size=0.2)

        train_data.to_csv(self.train_data_path,index=False)
        test_data.to_csv(self.test_data_path,index=False)

        return self.train_data_path,self.test_data_path