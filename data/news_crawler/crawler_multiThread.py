import pandas as pd
import numpy as np
from progressbar import *
import re
import time
import requests
import threading
from queue import Queue
from threading import Thread
from goose3 import Goose

# run function
def run(in_q, config):
    global num, processed_num, title_list

    widgets = ['progressing：', Percentage(), ' ', Bar('='), ' ', Timer(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=num).start()

    with Goose(config) as g:
        while in_q.empty() is not True:

            twe_info = in_q.get()
            twe_id = twe_info[0]
            url = twe_info[1][0]

            in_q.task_done()
            processed_num += 1
            pbar.update(processed_num)
            try:
                article = g.extract(url=url)
            except Exception as e:
                continue
            else:
                title = article.title
                content = article.cleaned_text
                if (title == '') or (len(title) <= 15) or (content == '') or ('404' in title):
                    continue
                else:
                    title_list.append((twe_id, title))
        pbar.finish()

# filter out the url in tweets
def getUrl(twe_text, pattern):
    url = re.findall(pattern, twe_text)
    if len(url) != 0:
        return url


if __name__=='__main__':
    # load the data
    twe_text = pd.read_csv('/media/yuting/TOSHIBA EXT/retweet/retweetstext.csv', \
                           header=None, \
                           names=['id_retweet', 'text', 'date', 'user_re', 'user_orig', 'id_twe'], \
                           sep='#1#8#3#', \
                           skiprows=1000, nrows=1000)
    # process pandas df
    twe_info = list(zip(twe_text['id_retweet'], twe_text['text']))
    # define the pattern to filer urls in tweets
    pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    # extract the urls
    twe_url = []
    for ele in twe_info:
        url = getUrl(ele[1], pattern)
        if url is not None:
            twe_url.append((ele[0], url))

    # set c
    config = {'strict':True, 'http_timeout':5} # turn of strict exception handling, set http timeout in seconds
    title_list = []

    # set mutli threading
    queue = Queue()
    for i in range(len(twe_url)):
        queue.put(twe_url[i])
    print('queue 开始大小 %d' % queue.qsize())

    num = queue.qsize()
    processed_num = 0

    for index in range(10):
        thread = Thread(target=run, args=(queue, config,))
        thread.daemon = True  # 随主线程退出而退出
        thread.start()

    queue.join()

    df = pd.DataFrame(title_list)
    df.to_csv("test_multi.csv", sep='\t', header=None, index=None, encoding='utf8')
