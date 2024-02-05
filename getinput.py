import numpy as np
import pandas as pd
from MCM.decorrelate import decorrelate
from MCM.expdecay import expdecay

data = pd.read_csv("C:/Users/。/Desktop/mc/data/data3.csv")

file_path = "C:/Users/。/Desktop/mc/data/output.txt"
score = pd.read_csv(file_path, header=None, names=["score1", "score2"])
score = score.score2 - score.score1


dec = decorrelate(score, data.serve)
data["flow"] = dec.decorr(score, data.serve)
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

for i in range(0, len(accident), 2):
    data[accident[i][3:]] = expdecay(
        data.match_id, data[accident[i]], data[accident[i + 1]]
    )
    data = data.drop(accident[i : i + 2], axis=1)

inputdata = data[
    [
        "match_id",
        "flow",
        "p1_distance_run",
        "p2_distance_run",
        "ace",
        "winner",
        "double_fault",
        "unf_err",
        "net_pt",
        "net_pt_won",
        "break_pt",
        "break_pt_missed",
    ]
]
inputdata.loc[:, "p1_distance_run"] = (
    inputdata["p1_distance_run"] - inputdata["p1_distance_run"].mean()
) / inputdata["p1_distance_run"].std()
inputdata.loc[:, "p2_distance_run"] = (
    inputdata["p2_distance_run"] - inputdata["p2_distance_run"].mean()
) / inputdata["p2_distance_run"].std()
inputdata.loc[:, "p1_distance_run"] = inputdata.groupby("match_id")[
    "p1_distance_run"
].shift(1, fill_value=0)
inputdata.loc[:, "p2_distance_run"] = inputdata.groupby("match_id")[
    "p2_distance_run"
].shift(1, fill_value=0)
inputdata.to_csv("C:/Users/。/Desktop/mc/data/inputdata.csv", index=False)
