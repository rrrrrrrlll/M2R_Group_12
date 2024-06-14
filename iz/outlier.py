import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def find_otl(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    outliers = []
    for n in data:
        if n > q3 + 2.5 * iqr or n < q1 - 2.5 * iqr:
            outliers.append(n)
    return outliers

def plot_outliers(df):

    df1 = df.T
    for i in range(len(df1)):
        data = df1.iloc[i]
        sns.boxplot(data)
        plt.show()

        outliers = find_otl(data)

        sns.histplot(outliers)
        plt.show()
        print(f"#outliers:{len(outliers)}")

def find_otl_index(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    outliers = []
    for n in range(len(data)):
        if data[n] > q3 + 2.5 * iqr or data[n] < q1 - 2.5 * iqr:
            outliers.append(n)
    return outliers

def freq_of_otl(df):
    otl_ind = []
    for i in range(len(df)):
        otl_seq += find_otl_index(df.iloc[i])
    
    sns.histplot(otl_ind)
    plt.show()

def remove_otl(df):
    """
    given a df, remove its outliers, i.e. n not in [q1 - 3 * iqr, q3 + 3 * iqr]
    """
    df1 = df
    otl_ind = set()
    for i in range(len(df.columns)):
        otl_ind |= set(find_otl_index(df1.iloc[:,i]))
    return df1.drop(otl_ind)
