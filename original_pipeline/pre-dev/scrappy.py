# -*- coding: utf-8 -*-
​
import requests
from bs4 import BeautifulSoup
import csv
​
def rss_to_dict(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.content, features='xml')
        
        return (soup)
    except Exception as e:
        print('The scraping job failed. See exception: ')
        print(e)
        
response = rss_to_dict('https://www.wgbh.org/news/term/boston-university-news-project.rss')
items = response.select('channel > item')
​
data = []
​
for item in items:
    title = (item.find('title')).text
    link = item.find('link').text
    description = item.find('description').text
    content = item.find('content:encoded').text
    category = item.find('category').text
    pubDate = item.find('pubDate').text
    
    mydict = {
        'title': title,
        'link': link,
        'description': description,
        'content': content,
        'category': category,
        'pubDate': pubDate
    }
    
    data.append(mydict)
    
lines = open('mycsvfile.csv', 'r').read()    
​
with open('mycsvfile.csv', 'a') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, data[0].keys())
    #w.writeheader()
    for item in data:
        print (item['title'], item['title'] in lines)
        w.writerow(item)