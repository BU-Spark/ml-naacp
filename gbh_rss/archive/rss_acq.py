#rss_acq.py
#aidan gomez ethan springer sahanna kowshik DEC 2022

#handles acquiring & formatting RSS data
#modified from scrappy.py in /predev

import requests
from bs4 import BeautifulSoup
import csv
from os.path import exists

import hashlib
import pandas as pd
import time

class rss_acquisition:
    def __init__(self):
        #can be reconfigured to work with different rss feeds
        self.rss_url = 'https://www.wgbh.org/news/term/boston-university-news-project.rss'
        self.rss_feed_path = 'temp/wgbh_rss_stored.csv'

        #do not change, these are modified during runtime

        #list of dictionaries
        self.data = None

        #any feed items loaded from csv in dataframe format
        self.loaded_feed = None

    
    #request and return RSS feed from url
    def rss_request(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
            r = requests.get(url, headers=headers)
            soup = BeautifulSoup(r.content, features='xml')
            return (soup)
        except Exception as e:
            print('> Err: The scraping job failed. See exception: ')
            print(e)
            return("")

    #parse scraped RSS feed
    def rss_parse(self):
        try:      
            response = self.rss_request(self.rss_url)
            items = response.select('channel > item')
            data = []
            mydict = {}
            for item in items:

                title = (item.find('title')).text
                link = item.find('link').text
                description = item.find('description').text
                content = item.find('content:encoded').text
                category = item.find('category').text
                pubDate = item.find('pubDate').text
                #print(type(title))
                #print(type(pubDate))
                mydict = {
                    'title': title,
                    'link': link,
                    'description': description,
                    'content': content,
                    'category': category,
                    'pubDate': pubDate, 
                    'UID': str(hashlib.md5( (str(title) + str(pubDate) ).encode()).hexdigest()) #UID of file
                }

                if(mydict!=None):
                    data.append(mydict)


            if(self.data!=None):
                self.data = data + self.data
            else: 
                self.data = data
            return(self.data)
        except Exception as e:
            print('> Err: The parsing job failed. See exception: ')
            print(e)

    #drop duplicates
    def remove_duplicates(self):
        df = pd.DataFrame.from_records(self.data).drop_duplicates(keep='first')
        self.data = df.to_dict('records')

    #save feed to file - using same file each time 
    #2-22-2023 I was having issues getting this to work - specifically getting it to open the file and then writing new articles to it
    def saveFeed2(self):
        try:
            if(self.data!=None):
                file_exists = exists(self.rss_feed_path)
                if(file_exists):
                    df = pd.read_csv(self.rss_feed_path)
                    acquired_df = pd.DataFrame.from_records(self.data)
                    ans = pd.concat([df, acquired_df]).drop_duplicates(keep='first')
                    print("> Saving to pre-existing RSS records: ", self.rss_feed_path)
                    ans.to_csv(self.rss_feed_path, sep='\t', encoding='utf-8')
                else:
                    pd.DataFrame.from_records(self.data).to_csv(self.rss_feed_path, sep='\t', encoding='utf-8')
            else:
                print("> Err: Requested save without any data")
        except Exception as e:
            print('> Err: The save job failed. See exception: ')
            print(e)

    # save feed to new file each time 
    def saveFeed(self):
        try:
            if(self.data!=None):
                # create new file path 
                print("> Saving feed...")
                new_path = "/Users/mvoong/Desktop/temp/" + str(time.time()).split(".")[0] + ".csv"
                df = pd.DataFrame.from_records(self.data)
                df.to_csv(new_path, encoding='utf-8')
                print(new_path)
            else:
                print("> Err: Requested save without any data")
        except Exception as e:
            print('> Err: The save job failed. See exception: ')
            print(e)

    #load from file into a dataframe, save into <self.loaded_feed>
    def loadFeedFromFile(self):
        try:
            file_exists = exists(self.rss_feed_path)
            if(file_exists):
                df = pd.read_csv (self.rss_feed_path)
                self.loaded_feed = df
                return(self.loaded_feed)
            else:
                print("> Err: Requested load of saved RSS feed that does not exist")
                return(False)
        except Exception as e:
            print('> Err: The load job failed. See exception: ')
            print(e)

    #return acquired data as dataframe
    def acquiredToDF(self):
        if(self.data!=None and self.data!=""):
            return(pd.DataFrame.from_records(self.data))
        else:
            return(None)
