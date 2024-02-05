import pandas as pd
import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
from MCM.lstm import Attention
from keras.models import load_model
from MCM.decorrelate import decorrelate
from MCM.expdecay import expdecay


def save(obj, dir):
    with open(dir, "wb") as file:
        pkl.dump(obj, file)


def load(dir):
    with open(dir, "rb") as file:
        return pkl.load(file)


class Monte:
    def __init__(self, model):
        self.population = pd.read_csv("C:/Users/。/Desktop/mc/data/data3.csv")
        self.population.loc[:, "p1_distance_run"] = (
            self.population["p1_distance_run"]
            - self.population["p1_distance_run"].mean()
        ) / self.population["p1_distance_run"].std()
        self.population.loc[:, "p2_distance_run"] = (
            self.population["p2_distance_run"]
            - self.population["p2_distance_run"].mean()
        ) / self.population["p2_distance_run"].std()
        self.model = model
        incident = "ace", "winner", "double_fault", "unf_err", "net_pt", "break_pt"
        for i in incident:
            exec(
                f"""self.{i}rate = (
                    (self.population.p1_{i} + self.population.p2_{i}).sum()
                    / len(self.population)
                    / 2
                    )"""
            )
        self.npt_wonrate = (
            self.population.p1_net_pt_won + self.population.p2_net_pt_won
        ).sum() / (self.population.p1_net_pt + self.population.p2_net_pt).sum()
        self.bpt_missrate = (
            self.population.p1_break_pt_missed + self.population.p2_break_pt_missed
        ).sum() / (self.population.p1_break_pt + self.population.p2_break_pt).sum()

    def prepare(self, length, copys, match_data):
        self.match_data = match_data
        self.length = length
        self.copyl = copys
        # get distance
        self.distance = (
            self.population.sample(n=length * copys, replace=True)
            .loc[:, ("p1_distance_run", "p2_distance_run")]
            .reset_index(drop=True)
        )
        # get incident
        self.ace = pd.Series(
            np.random.choice(
                [0, 1], size=2 * length * copys, p=[1 - self.acerate, self.acerate]
            ),
            name="ace",
        )
        self.winner = pd.Series(
            np.random.choice(
                [0, 1],
                size=2 * length * copys,
                p=[1 - self.winnerrate, self.winnerrate],
            ),
            name="winner",
        )
        self.double_fault = pd.Series(
            np.random.choice(
                [0, 1],
                size=2 * length * copys,
                p=[1 - self.double_faultrate, self.double_faultrate],
            ),
            name="double_fault",
        )
        self.unf_err = pd.Series(
            np.random.choice(
                [0, 1],
                size=2 * length * copys,
                p=[1 - self.unf_errrate, self.unf_errrate],
            ),
            name="unf_err",
        )
        self.net_pt = pd.Series(
            np.random.choice(
                [0, 1],
                size=2 * length * copys,
                p=[1 - self.net_ptrate, self.net_ptrate],
            ),
            name="net_pt",
        )
        self.break_pt = pd.Series(
            np.random.choice(
                [0, 1],
                size=2 * length * copys,
                p=[1 - self.break_ptrate, self.break_ptrate],
            ),
            name="break_pt",
        )

        change_mask = np.random.choice(
            [0, 1], size=length * copys * 2, p=[1 - self.npt_wonrate, self.npt_wonrate]
        )
        self.net_pt_won = pd.Series(
            np.where(
                (self.net_pt.to_numpy() == 1) & (change_mask == 0),
                0,
                self.net_pt.to_numpy(),
            ),
            name="net_pt_won",
        )
        change_mask = np.random.choice(
            [0, 1], size=length * copys * 2, p=[1 - self.npt_wonrate, self.npt_wonrate]
        )
        self.break_pt_missed = pd.Series(
            np.where(
                (self.net_pt.to_numpy() == 1) & (change_mask == 0),
                0,
                self.net_pt.to_numpy(),
            ),
            name="net_pt_won",
        )
        try:
            self.copys
        except AttributeError:
            self.copys = []
        self.origin = self.population[
            self.population.match_id == self.match_data.match_id[0]
        ]

    def generate(self):
        for i in range(self.copyl):
            p1_ace = self.ace[
                self.length * 2 * i : self.length * (2 * i + 1)
            ].reset_index(drop=True)
            p2_ace = self.ace[
                self.length * (2 * i + 1) : self.length * (2 * i + 2)
            ].reset_index(drop=True)
            p1_winner = self.winner[
                self.length * 2 * i : self.length * (2 * i + 1)
            ].reset_index(drop=True)
            p2_winner = self.winner[
                self.length * (2 * i + 1) : self.length * (2 * i + 2)
            ].reset_index(drop=True)
            p1_double_fault = self.double_fault[
                self.length * 2 * i : self.length * (2 * i + 1)
            ].reset_index(drop=True)
            p2_double_fault = self.double_fault[
                self.length * (2 * i + 1) : self.length * (2 * i + 2)
            ].reset_index(drop=True)
            p1_unf_err = self.unf_err[
                self.length * 2 * i : self.length * (2 * i + 1)
            ].reset_index(drop=True)
            p2_unf_err = self.unf_err[
                self.length * (2 * i + 1) : self.length * (2 * i + 2)
            ].reset_index(drop=True)
            p1_net_pt = self.net_pt[
                self.length * 2 * i : self.length * (2 * i + 1)
            ].reset_index(drop=True)
            p2_net_pt = self.net_pt[
                self.length * (2 * i + 1) : self.length * (2 * i + 2)
            ].reset_index(drop=True)
            p1_net_pt_won = self.net_pt_won[
                self.length * 2 * i : self.length * (2 * i + 1)
            ].reset_index(drop=True)
            p2_net_pt_won = self.net_pt_won[
                self.length * (2 * i + 1) : self.length * (2 * i + 2)
            ].reset_index(drop=True)
            p1_break_pt = self.break_pt[
                self.length * 2 * i : self.length * (2 * i + 1)
            ].reset_index(drop=True)
            p2_break_pt = self.break_pt[
                self.length * (2 * i + 1) : self.length * (2 * i + 2)
            ].reset_index(drop=True)
            p1_break_pt_missed = self.break_pt_missed[
                self.length * 2 * i : self.length * (2 * i + 1)
            ].reset_index(drop=True)
            p2_break_pt_missed = self.break_pt_missed[
                self.length * (2 * i + 1) : self.length * (2 * i + 2)
            ].reset_index(drop=True)
            p1_dist = self.distance.p1_distance_run[
                self.length * i : self.length * (i + 1)
            ].reset_index(drop=True)
            p2_dist = self.distance.p2_distance_run[
                self.length * i : self.length * (i + 1)
            ].reset_index(drop=True)
            df = pd.concat(
                [
                    p1_dist,
                    p2_dist,
                    p1_ace,
                    p2_ace,
                    p1_winner,
                    p2_winner,
                    p1_double_fault,
                    p2_double_fault,
                    p1_unf_err,
                    p2_unf_err,
                    p1_net_pt,
                    p2_net_pt,
                    p1_net_pt_won,
                    p2_net_pt_won,
                    p1_break_pt,
                    p2_break_pt,
                    p1_break_pt_missed,
                    p2_break_pt_missed,
                ],
                axis=1,
                keys=[
                    "p1_distance_run",
                    "p2_distance_run",
                    "p1_ace",
                    "p2_ace",
                    "p1_winner",
                    "p2_winner",
                    "p1_double_fault",
                    "p2_double_fault",
                    "p1_unf_err",
                    "p2_unf_err",
                    "p1_net_pt",
                    "p2_net_pt",
                    "p1_net_pt_won",
                    "p2_net_pt_won",
                    "p1_break_pt",
                    "p2_break_pt",
                    "p1_break_pt_missed",
                    "p2_break_pt_missed",
                ],
            )
            cat = pd.concat([self.origin[df.columns], df], axis=0).reset_index(
                drop=True
            )
            cat["match_id"] = self.match_data.match_id[0]
            accident = [
                "p1_ace",
                "p2_ace",
                "p1_winner",
                "p2_winner",
                "p1_double_fault",
                "p2_double_fault",
                "p1_unf_err",
                "p2_unf_err",
                "p1_net_pt",
                "p2_net_pt",
                "p1_net_pt_won",
                "p2_net_pt_won",
                "p1_break_pt",
                "p2_break_pt",
                "p1_break_pt_missed",
                "p2_break_pt_missed",
            ]
            for j in range(0, len(accident), 2):
                cat[accident[j][3:]] = expdecay(
                    cat.match_id, cat[accident[j]], cat[accident[j + 1]]
                )
                cat = cat.drop(accident[j : j + 2], axis=1)

            cat.loc[:, "p1_distance_run"] = cat.groupby("match_id")[
                "p1_distance_run"
            ].shift(1, fill_value=0)
            cat.loc[:, "p2_distance_run"] = cat.groupby("match_id")[
                "p2_distance_run"
            ].shift(1, fill_value=0)
            cat = cat.drop("match_id", axis=1)
            cat["flow"] = self.match_data.reset_index(drop=True)["flow"]
            cat = cat.fillna(0)
            cat["flow"] = cat.flow.shift(fill_value=0)
            self.copys.append(cat)

    def clone(self, length, copys, data):
        self.prepare(length, copys, data)
        self.generate()

    def predict(self):
        self.perf = []
        for l in range(len(self.copys)):
            perf = 0
            for i in range(self.length):
                X = self.copys[l].iloc[: -self.length + i, :]
                newperf = self.model(X.to_numpy())[-1, 0].numpy()
                perf += newperf
                print(-self.length + i, newperf)
                self.copys[l].iloc[-self.length + i, -1] = newperf
            self.perf.append(perf)


if __name__ == "__main__":
    model = load_model(
        "C:/Users/。/Desktop/mc/cache/my_model.h5",
        custom_objects={"Attention": Attention},
    )
    data = pd.read_csv("C:/Users/。/Desktop/mc/data/inputdata.csv")
    matchid = 1301
    data = data[data.match_id == matchid]
    mont = Monte(model)
    mont.prepare(4, data)
    mont.generate()
    mont.predict()
