import pandas as pd
import numpy as np
class stochasitic:
    def __init__(self) -> None:
        pass    
    def train(self,df):
        self.serve_no=df['serve_no']
        self.point_victor=df['point_victor']
        self.P1=self.point_victor[self.serve_no==1].mean()
        self.P2=self.point_victor[self.serve_no==2].mean()
    def predict(self):
        mapping={1:self.P1,2:self.P2}
        probability=self.serve_no.map(mapping)
        randomvalues=np.random.uniform(1,2,len(probability))
        randomseries=pd.Series(randomvalues,index=probability.index)
        self.victor_predicted=(randomseries>probability).apply((lambda x: 2 if x else 1))
    def accuracy(self):
        correct_predictions = (self.victor_predicted == self.point_victor).sum()
        return correct_predictions/len(self.point_victor)

