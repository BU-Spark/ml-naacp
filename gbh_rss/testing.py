import rss_acq


pipli = rss_acq.rss_acquisition()
pipli.rss_request(pipli.rss_url)
temp = pipli.rss_parse()
pipli.saveFeed()

