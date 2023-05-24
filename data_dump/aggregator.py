import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

print("**** Running Data Aggregation ****")
csv_files = [f for f in os.listdir() if 'agg' not in f]
print(csv_files)

import pandas as pd

data = pd.concat([pd.read_csv(f_csv) for f_csv in csv_files])

data.drop(["Unnamed: 0"],axis=1,inplace=True)
data.drop_duplicates(inplace=True)
data.to_csv("aggregated.csv")

self_selected = pd.read_csv('agg_self_selected_covid_neutral_sentiment.csv',engine='python',sep='\r\n')
data = pd.concat([data,self_selected])
data.drop_duplicates(inplace=True)

data.to_csv("aggregated.csv")