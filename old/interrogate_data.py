import torch
import prep_data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")

train_csv_path = "C:/Users/joshc/OneDrive/Documents/01 Trying too hard/Machine Learning and AI/Kaggle/titanic/Datasets/train.csv"

train_data_df = pd.read_csv(train_csv_path)
survived_only = train_data_df[train_data_df["Survived"] == 1].copy()


def display_na_counts(df):
    print(df.isna().sum())
    
def bucketize_by_key(df, key: str, n_buckets: int =5):
    bucket_key = f"Bucket {key.capitalize()}"
    df[bucket_key] = pd.cut(train_data_df[key], n_buckets)
    return df

def count_buckets(df, key: str, n_buckets: int =5):
    bucket_key = f"Bucket {key.capitalize()}"
    bucket_count = df.groupby(bucket_key, dropna=False).count()["PassengerId"]
    bucket_count["None"] = df[bucket_key].isna().sum()
    return bucket_count

def double_hist_plot():
    """Bucketises data and then plots bar plots. Bar plot of own-made 
    buckets is more customisable than sns.displot()"""
    n_buckets = 10
    bucketize_by_key(train_data_df, "Age", n_buckets=n_buckets)
    bucketize_by_key(survived_only, "Age", n_buckets=n_buckets)
    age_count = count_buckets(train_data_df, "Age",n_buckets=n_buckets)
    age_count_survived = count_buckets(survived_only, "Age", n_buckets=n_buckets)
    age_survival_rate = age_count_survived/age_count
    num_df = pd.DataFrame(age_count.values)[:-1]
    
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(5.5, 3.5),
                        constrained_layout=True)
    fig.suptitle("Survival rates in age buckets, and quantity in each bucket ")
    axs[1].set_xticks(age_survival_rate.index[:-1], rotation=45)
    sns.barplot(x=age_survival_rate.index, y=age_survival_rate.values,ax=axs[0])
    sns.scatterplot(data=num_df, ax=axs[1], legend=False)
    plt.show()

def regular_hist_plot():
    sns.displot(data=train_data_df, x="Age", kde=True)
    sns.displot(data=survived_only, x="Age", kde=True)
    plt.show()

double_hist_plot()