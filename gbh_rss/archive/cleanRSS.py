import pandas as pd
import os
import glob



def combine_csvs():
    # combine all of the scraped csvs  
    dir_path = "../dev/temp/"
    directory = os.listdir(dir_path)
    filenames = glob.glob(os.path.join(dir_path, "*.csv"))
    df = pd.concat((pd.read_csv(f) for f in filenames), ignore_index=True)
    df = df.drop(df.columns[0], axis=1) # remove the first column which is just the index  
    df = df.drop_duplicates() # removing duplicates

    return df

def add_to_full_feed(df):
    # combining the new scraped csvs with what has already been collected in gbh-rss-feed.csv
    full_feed = pd.read_csv("gbh-rss-feed.csv")
    updated_feed = pd.concat([full_feed, df], ignore_index=True)
    updated_feed.to_csv("gbh-rss-feed.csv")
    

