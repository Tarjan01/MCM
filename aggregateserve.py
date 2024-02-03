import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import pickle as pkl
import matplotlib.pyplot as plt


def save(obj, dir):
    with open(dir, "wb") as file:
        pkl.dump(obj, file)


def load(dir):
    with open(dir, "rb") as file:
        return pkl.load(file)



def deal_serve(df:pd.DataFrame):
    ''' this is used to aggregate features concerning serving'''
    if 'serve' in df.columns:
        print('already modified')
        return df
    else:
        df=df.drop('return_depth',axis=1)
        # standardize the speed_mph
        mean = df['speed_mph'].mean()
        std = df['speed_mph'].std()
        df['speed_mph'] = (df['speed_mph'] - mean) / std

        # see the missing gap
        # plt.plot(df.speed_mph)
        # plt.title("missing gap of speed_mph")
        # plt.show()


        data = df.drop(range(81, 569), axis=0)
        # fill the residual speed with mean
        data.speed_mph = data.speed_mph.fillna(data.speed_mph.mean())

        # smooth the speed distribution with kde
        speed = data[data.point_victor == data.server].speed_mph
        kde = gaussian_kde(
            speed,
        )
        x_grid = np.linspace(speed.min() - 1, speed.max() + 1, 1000)
        kde_values = kde.evaluate(x_grid)
        speed1 = data[data.point_victor != data.server].speed_mph
        kde = gaussian_kde(
            speed1,
        )
        kde_values1 = kde.evaluate(x_grid)

        bestspeed = x_grid[(kde_values - kde_values1).argmax()]
        data["serve"] = (data.server * 2 - 3) * np.exp(-np.abs(data.speed_mph - bestspeed))

        # drop missing serve_width
        data = data.dropna(how="any")


        # assign number for serve_width (include B or not)
        def chech_char(s, char):
            if s is np.nan:
                return s
            else:
                return False if char in s else True

        data.serve_width = data.serve_width.apply(chech_char, args=("B",))
        data.loc[data["serve_width"], "serve"] *= 2
        df = pd.concat([df, data.serve], axis=1)
        negmean = df[df.serve < 0].serve.mean()
        posmean = df[df.serve > 0].serve.mean()
        df["server"] = df["server"].replace({1:negmean,2:posmean})
        df["serve"] = df["serve"].fillna(df["server"])
        filtered_column = [item for item in df.columns if "serve" not in item]
        filtered_column.extend(['serve', 'serve_no'])
        new_index = pd.Index(filtered_column)
        new_index=new_index.drop(['speed_mph'])
        return df[new_index]


if __name__=='__main__':
    data = pd.read_csv("C:/Users/。/Desktop/mc/data/data2(1).csv")
    data1=deal_serve(data)
    data1.to_csv("C:/Users/。/Desktop/mc/data/data3.csv")