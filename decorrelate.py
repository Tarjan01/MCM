import pandas as pd
import numpy as np


class decorrelate:
    def __init__(self, main: pd.Series, sub: pd.Series):
        self.mainstd = main.std()
        self.substd = sub.std()
        self.cor = main.corr(sub)
        self.intercept = (main - self.cor * self.mainstd / self.substd * sub).mean()

    def decorr(self, main: pd.Series, sub: pd.Series):
        return (main - self.cor * self.mainstd / self.substd * sub - self.intercept) / (
            self.mainstd * np.sqrt(1 + self.cor**2)
        )

    def inverse(self, res: pd.Series, sub: pd.Series):
        return (
            res * (self.mainstd * np.sqrt(1 + self.cor**2))
            + self.intercept
            + self.cor * self.mainstd / self.substd * sub
        )




if __name__ == "__main__":
    score = pd.read_csv("C:/Users/。/Desktop/mc/data/scoreDenoised_final.csv")
    data = pd.read_csv("C:/Users/。/Desktop/mc/data/data3.csv")
    data["diffe"] = score["player2"] - score["player1"]
    data["diff_diffe"] = data.groupby("match_id")["diffe"].transform(
        lambda x: x.diff().fillna(x.iloc[0])
    )
    data["diff_diffe"] = (data["diff_diffe"] - data["diff_diffe"].mean()) / data[
        "diff_diffe"
    ].std()
    print(data.serve.corr(data.diff_diffe))
    # decorelate(data.serve,)
