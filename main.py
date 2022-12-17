from sqlalchemy.orm import Session
from models import NewsStream, News, Base, Corpus, Subcorpus, Analyse, Cluster, Match
from sqlalchemy import create_engine, select
import feedparser
import datetime
import hashlib
import time
from sklearn.cluster import OPTICS
import datetime
import numpy as np
from bs4 import BeautifulSoup
import urllib.request

def engine_factory(args):
    engine = create_engine(args.database, echo=args.debug, future=True)
    return engine

def init(args):
    engine = engine_factory(args)
    Base.metadata.create_all(engine)

def importrss(args):
    engine = engine_factory(args)
    import csv
    with open(args.file, 'r') as file:
        csvFile = csv.DictReader(file, delimiter=',')
        with Session(engine) as session:
            for line in csvFile:    
                news = NewsStream()
                for arg in line.items():
                    setattr(news, arg[0], arg[1])
                session.add(news)
            session.commit()

def sync(args):
    engine = engine_factory(args)
    with Session(engine) as session:
        for stream in session.scalars(select(NewsStream)):
            sync_stream(stream, session)

def sync_stream(stream, session):
    feed = feedparser.parse(stream.url)
    for entry in feed.entries:
        news = parse_entrie(entry)
        if news is None:
            continue
        news.source_id = stream.id
        try:
            session.add(news)
            session.commit()
            print(news)
        except Exception as e:
            print(e)
            session.rollback()
            continue



def parse_entrie(entry):
    if 'title' not in entry:
        #title is one of the most common header, if not present something is fucked
        return None

    title = entry.title
    description_bs = BeautifulSoup(getattr(entry, 'summary', title))
    description = description_bs.get_text()
    guid = getattr(entry, 'id', entry.link)
    pubdate = getattr(entry, 'updated_parsed', None)
    pubdate = getattr(entry, 'created_parsed', pubdate)
    pubdate = getattr(entry, 'published_parsed', pubdate)
    if pubdate is None:
        pubdate = datetime.datetime.now()
    else:
        pubdate = datetime.datetime.fromtimestamp(time.mktime(pubdate))

    media = None
    for elink in entry.links:
        if elink.type.startswith("image/"):
            media = elink.href
    if media is None:
        img = description_bs.find("img")
        if img is not None:
            media = img.get("link")

    link_as_id = entry.link == guid
    try:
        link = get_cannonical_url(entry.link)
    except urllib.error.HTTPError:
        # if we are not able to see it, safer not to get it lol
        return None
    except urllib.error.URLError:
        # if we are not able to see it, safer not to get it lol
        return None
    if link_as_id:
        guid = link

    return News(
        guid=guid,
        title=title,
        description=description,
        link=link,
        pubdate=pubdate,
        media=media,
    )

def get_cannonical_url(url):
    best_link = url
    opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler)
    with opener.open(url) as response:
        best_link = response.url
        soup=BeautifulSoup(response.read(), "html.parser")
        link=soup.find("link", rel="canonical")
        if link is not None:
            best_link = link["href"]
    return clean_link(best_link)

def clean_link(url):
    return url.split('#')[0].split('?')[0]

def analyse(args):
    engine = engine_factory(args)
    with Session(engine) as session:
        corpus = Corpus.from_iter(session.scalars(select(News)))
        subset = Subcorpus(corpus, lambda x: x.pubdate > datetime.datetime.now() - datetime.timedelta(days=1))
        dataset = [vec for vec in subset.vectorize()]
        clustering = OPTICS(min_samples=3, metric='cosine').fit(dataset)
        cluster = {}
        for i in range(len(clustering.labels_)):
            if clustering.labels_[i] == -1:
                continue
            if clustering.labels_[i] not in cluster:
                cluster[clustering.labels_[i]] = []
            cluster[clustering.labels_[i]].append((subset[i], dataset[i]))

        analyse = Analyse()
        session.add(analyse)
        for k, v in cluster.items():
            cltr = Cluster(analyse=analyse)
            session.add(cltr)

            avg = np.average([t[1] for t in v], axis=0)
            v.sort(key=lambda x: np.linalg.norm(avg - x[1]))
            for t in v:
                distance = np.linalg.norm(avg - t[1])
                _match = Match(cluster=cltr, news=t[0], distance=distance)
                session.add(_match)
        session.commit()
            

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        prog='news aggregator',
        description='suck some dicks to get the juce',
    )
    parser.add_argument('--database', type=str, default="sqlite:////home/droper/news.db")
    parser.add_argument('--debug', action='store_const', default=False, const=True)
    subparsers = parser.add_subparsers()
    parser_init = subparsers.add_parser('init')
    parser_init.set_defaults(func=init)
    parser_import = subparsers.add_parser('import')
    parser_import.add_argument('--file', type=str) 
    parser_import.set_defaults(func=importrss)
    parser_sync = subparsers.add_parser('sync')
    parser_sync.set_defaults(func=sync)
    parser_analyse = subparsers.add_parser('analyse')
    parser_analyse.set_defaults(func=analyse)
    args = parser.parse_args()
    args.func(args)
