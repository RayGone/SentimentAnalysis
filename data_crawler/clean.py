import pandas as pd
import os
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.realpath(__file__)))

csv_files = [f for f in os.listdir() if '.csv' in f and 'clean' not in f]
print(csv_files)

agg_data = pd.DataFrame(columns=['text'])
for cf in tqdm(csv_files):
    data = pd.read_csv(cf)
    data = data[['text']]
    data = data.drop_duplicates()

    ## remove news initial text
    def removeNewsInitial(x):    
        txt = x['text'].split("।")
        if len(txt) > 1 and len(txt[0]) <= 25:
            x['text'] = "।".join(txt[1:])
        return x

    data = data.apply(lambda x: removeNewsInitial(x),axis=1)
    ## end 
    # print("Saving Individual file")
    data.to_csv("cleaned_{}".format(cf))
    
    ### Collective data
    data['source'] = cf
    agg_data = pd.concat([agg_data,data])
    
print(agg_data.shape)
print("Saving Collective Data To File: NepCovNews.csv")
agg_data.to_csv("NepCovNews.csv")

# import datasets
# datasets.Dataset.from_pandas(agg_data).push_to_hub("raygx/NepCovidHealthNews",token='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')