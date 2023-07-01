import os
import pandas as pd

os.chdir(os.path.dirname(os.path.realpath(__file__)))

print("**** Running Data Aggregation ****")
csv_files = [f for f in os.listdir() if 'agg' not in f.lower() and 'tweets' not in f]
print(csv_files)

### Task 1
if False: ## fixing tweets crawled data
    data = pd.concat([pd.read_csv(x,engine='python',sep='\r\n') for x in csv_files if 'tweets' in x])
    data = data.drop_duplicates()
    data.to_csv("cleaned_aggregated_tweets.csv")

## Task 2
if True:    
    data = pd.concat([pd.read_csv(f_csv)[['text']] for f_csv in csv_files])
    print("Before Drop Duplicate:",data.shape)
    data.drop_duplicates(inplace=True)
    print("After Drop Duplicate:",data.shape)
    data.to_csv("aggregated_common.csv")
    print("Complete")
# self_selected = pd.read_csv('agg_self_selected_covid_neutral_sentiment.csv',engine='python',sep='\r\n')
# data = pd.concat([data,self_selected])
# data.drop_duplicates(inplace=True)

# data.to_csv("aggregated.csv")

if True: ## Task 3
    print("#Task 3")
    # काेराेना भाइरस मास्क कोभिड महामारी -- मृत्यु , ज्यान ग (गुमाउने, गएको) , आरोप
    data = pd.read_csv("aggregated_common.csv")[['text']]
    corona = data[data['text'].str.contains("काेराेना")]
    virus = data[data['text'].str.contains("भाइरस")]
    mask = data[data['text'].str.contains("मास्क")]
    covid = data[data['text'].str.contains("कोभिड")]
    pandemic = data[data['text'].str.contains("महामारी")]
    
    data = pd.concat([corona, virus, mask, covid, pandemic])
    print(data.shape) 
    
    data = data[~data['text'].str.contains('मृत्यु')]
    data = data[~data['text'].str.contains('ज्यान ग')]
    data = data[~data['text'].str.contains('आरोप')]
    print(data)
    data.to_csv("aggregated_common_cleaned.csv")