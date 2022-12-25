import pandas as pd
import numpy as np
import time
import datetime
import plotly.express as px
import plotly.graph_objects as go

pd.options.plotting.backend = "plotly"

import numpy as np

x0 = np.random.randn(500)
# Add 1 to shift the mean of the Gaussian distribution
x1 = np.random.randn(500) + 1

fig = go.Figure()
fig.add_trace(go.Histogram(x=x0))
fig.add_trace(go.Histogram(x=x1))

# Overlay both histograms
fig.update_layout(barmode="overlay")
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)
fig.show()
# This is a sample Python script.

import plotly.graph_objects as go

fig = go.Figure()
df = df[df["bedrooms"] < 12]
for nr_of_bedrooms in set(df["bedrooms"]):
    fig.add_trace(
        go.Violin(
            x=df["bedrooms"][df["bedrooms"] == nr_of_bedrooms],
            y=df["price"][df["bedrooms"] == nr_of_bedrooms],
            name=nr_of_bedrooms,
            box_visible=True,
            meanline_visible=True,
        )
    )


def plot_hist(col):
    fig = go.Figure()
    # fig.add_trace(go.Histogram(x=df[col]))
    fig.add_trace(go.Histogram(x=df[col], y=df["price"]))

    # Overlay both histograms
    fig.update_layout(barmode="overlay")
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    fig.show()


fig = go.Figure(data=go.Heatmap(z=[[1, 20, 30], [20, 1, 60], [30, 60, 1]]))
fig.show()
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f"Hi, {name}")  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    print_hi("PyCharm")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


def violinplot():
    fig = go.Figure()
    fig.update_layout(autosize=False, width=1500, height=1000)
    df = df[df["bedrooms"] < 12]
    for nr_of_bedrooms in set(df["bedrooms"]):
        fig.add_trace(
            go.Violin(
                x=df["bedrooms"][df["bedrooms"] == nr_of_bedrooms],
                y=df["price"][df["bedrooms"] == nr_of_bedrooms],
                name=nr_of_bedrooms,
                box_visible=True,
                meanline_visible=True,
            )
        )

    fig.show()


null_frac = 0.05
quot = 1 / null_frac
noisy_df = df
for col_name in df.columns:
    noisy_df = noisy_df.withColumn(
        col_name, when(abs(hash(col_name)) % quot != 0, col(col_name))
    )
noisy_df.show()
