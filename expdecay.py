import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def expdecay(match_id: pd.Series, acc1: pd.Series, acc2: pd.Series):
    def exponential_decay(impulse_series, decay_rate=1):
        accumulation = np.zeros(len(impulse_series))  # 初始化累积数组
        for i in range(len(impulse_series)):
            if impulse_series.iloc[i] == 1:  # 使用.iloc来访问序列
                for j in range(i, min(len(impulse_series), i + 20)):
                    accumulation[j] += np.exp(-decay_rate * (j - i))
        return accumulation

    def shift_group(series):
        return series.shift(1).fillna(0)

    df = pd.concat([match_id, acc1, acc2], axis=1)
    df["accumulation"] = df.groupby("match_id")[acc2.name].transform(
        exponential_decay
    ) - df.groupby("match_id")[acc1.name].transform(exponential_decay)
    shifted = df.groupby("match_id")["accumulation"].apply(shift_group)
    shifted.index = shifted.index.droplevel("match_id")
    return shifted


if __name__ == "__main__":
    data = pd.read_csv("C:/Users/。/Desktop/mc/data/data3.csv")
    data["accumulation"] = expdecay(
        data.match_id,
        pd.Series(np.zeros_like(data.p2_unf_err), name="empt"),
        data.p2_unf_err,
    )
    plt.figure(figsize=(20, 8))

    match_id = data.match_id.unique()[0]
    subset = data[data["match_id"] == match_id].iloc[:100]
    plt.plot(
        subset.index, subset["accumulation"], label=f"Accumulation for match {match_id}"
    )
    pulse_points = subset[subset["p2_unf_err"] == 1]
    plt.scatter(
        pulse_points.index,
        [0] * len(pulse_points),
        color="red",
        label=f"Pulse Points for match {match_id}",
        zorder=5,
    )

    plt.title(
        "Exponential Decay of Accumulation from p2_unf_err with Pulse Points on a match"
    )
    plt.xlabel("spot")
    plt.ylabel("Accumulation")
    plt.legend()
    plt.grid(True)
    plt.show()