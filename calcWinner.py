import pandas as pd
import numpy as np

def calWinner():
    data3 = pd.read_csv('./assets/doc/data3.csv', encoding='utf-8')
    winner = [0] * data3.shape[0]

    with open ('./assets/doc/winner.csv', 'w') as f:
        for i in range(data3.shape[0]):
            if(i != 0 and data3.match_id[i] == data3.match_id[i-1]):
                if(data3.p1_points_won[i] > data3.p1_points_won[i-1]):
                    winner[i] = 1
                else:
                    winner[i] = 2
            else:
                if(data3.p1_points_won[i] != 0):
                    winner[i] = 1
                else:
                    winner[i] = 2

    with open('./assets/doc/winner.csv', 'w') as f:
        for i in range(data3.shape[0]):
            print(f"{winner[i]}", file=f)

    
if __name__ == "__main__":
    pass