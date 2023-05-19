import requests
from lxml import html
import pandas as pd


def crawlEkantipur(save_output=True):
    api_point = "https://ekantipur.com/health/"+"{y}/{m}/{d}"#.format(y=2021,m=11,d=28)
    data = pd.DataFrame(columns=['text'])
    
    year = 2020
    month = 1
    date = 0
    
    while True:
        date = date+1     
        url = api_point.format(y=year,m=month,d=date)
        print("Fetching URL",url)
        page = requests.get(url)
        if page.history:
            date = 0
            year = year if month <12 else year+1
            if year == 2023:
                print("Done Collecting upto 2022")
                break
            
            month = 1 if month==12 else month+1
        page = page.content.decode("utf-8")
        tree = html.document_fromstring(page)
        

        elements = tree.find_class('logo-white')
        if not elements:
            print("No elements found")
            continue
        
        elements = tree.find_class("teaser")
        
        data = pd.concat([data,pd.DataFrame([x.find("p").text_content() for x in elements if x.find("p") is not None],columns=['text'])])
    
    data.to_csv("ekantipur.csv")    
    return data
    

def crawlImageKhabar(save_output=True):
    api_point = "https://www.imagekhabar.com/news/tag/कोरोना/"
    data = pd.DataFrame(columns=['text'])
    i=0
    while True:
        url = api_point+"{}/{}".format("page",i)
        print("Fetching URL",url)

        page = requests.get(url).content.decode("utf-8")
        tree = html.document_fromstring(page)

        elements = tree.find_class("uk-card-body")
        if not elements:
            break
        
        data = pd.concat([data,pd.DataFrame([x.find("p").text_content() for x in elements if x.find("p") is not None],columns=['text'])])
        
        i = i+1
        

    data.to_csv("imagekhabar.csv")
    return data


if(__name__ == "__main__"):
    print(crawlEkantipur())