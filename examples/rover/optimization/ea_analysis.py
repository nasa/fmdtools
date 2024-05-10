# -*- coding: utf-8 -*-
"""
Plots all fitness values of top performing individuals per generation
for each algorithm, i.e., CCEA, EA, and Random
"""

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dfea = pd.read_csv('results/rslt_ea.csv')
    dfrand = pd.read_csv('results/rslt_random.csv')
    # dfccea = pd.read_csv("ccea_pop.csv")
    dfccea = pd.read_csv("results/rslt_ccea.csv")

    print(dfea.head())
    print(dfrand.head())
    print(dfccea.head())

    # scatter plot
    ax = dfea.plot(x='time', y='EA Fitness Values',
                   c='r', kind='scatter', label='EA')
    dfrand.plot(x='time', y='Random Fitness Values', kind='scatter', ax=ax,
                c='g', label='Monte Carlo')
    dfccea.plot(x='time', y='CCEA Fitness Values', kind='scatter', ax=ax,
                c='b', label='CCEA')

    # set the title

    plt.ylabel("Objective Value")
    plt.xlabel("Computational Time (s)")
    plt.title("Search Performance Comparison")

    # show the plot
    plt.show()
