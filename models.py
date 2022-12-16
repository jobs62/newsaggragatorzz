from sqlalchemy import Column, ForeignKey, String, DateTime, Integer, Numeric
from sqlalchemy.orm import declarative_base, relationship
import re
from removeaccents import removeaccents
import snowballstemmer
import numpy as np
import bs4
from datetime import datetime

Base = declarative_base()

STOPWORDS = set([
    "a",
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "d",
    "dans",
    "de",
    "du",
    "elle",
    "en",
    "et",
    "eux",
    "il",
    "je",
    "la",
    "le",
    "les",
    "leur",
    "lui",
    "ma",
    "mais",
    "me",
    "meme",
    "mes",
    "moi",
    "mon",
    "ne",
    "nos",
    "notre",
    "nous",
    "on",
    "ou",
    "par",
    "pas",
    "pour",
    "qu",
    "que",
    "qui",
    "sa",
    "se",
    "ses",
    "son",
    "sur",
    "ta",
    "te",
    "tes",
    "toi",
    "ton",
    "tu",
    "un",
    "une",
    "vos",
    "votre",
    "vous",
    "a",
    "à",
    "l",
    "est",
    "y",
    "sont",
    "été",
    "où",
    "n",
])

class NewsStream(Base):
    __tablename__ = "news_stream"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String)
    display_name = Column(String)

    news = relationship('News')

    def __repr__(self):
        return f"NewsStream(id={self.id!r}, display_name={self.display_name!r},  url={self.url!r})"

class News(Base):
    __tablename__ = "news"

    guid = Column(String, primary_key=True, nullable=False)
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    link = Column(String, nullable=True)
    media = Column(String, nullable=True)
    pubdate = Column(DateTime, nullable=False)
    
    source_id = Column(Integer, ForeignKey("news_stream.id"))
    source = relationship(
        "NewsStream", cascade="all", back_populates="news"
    )

    def __repr__(self):
        return f"News(guid={self.guid!r}, title={self.title!r}, link={self.link!r}, description={self.description!r}, pubdate={self.pubdate!r}, media={self.media!r})"

    def __get_tf__(self):
        if hasattr(self, '__tf__'):
            # caching tf
            return self.__tf__
        self.__tf__ = {}
        self.__update_tf__(self.title)
        if self.description is not None and self.description != self.title:
            description = bs4.BeautifulSoup(self.description).get_text()
            self.__update_tf__(description)
        return self.__tf__

    def __update_tf__(self, txt: str):
        for feature in News.extract_feature(txt):
            if feature not in self.__tf__:
                self.__tf__[feature] = 0
            self.__tf__[feature] += 1

    @staticmethod
    def extract_feature(txt):
        stemmer = snowballstemmer.stemmer("french")
        for s in News.split_sentances(txt):
            single_feature = []
            for w in News.split_word(s):
                word = w.lower()
                if word in STOPWORDS:
                    continue
                feature = removeaccents.remove_accents(stemmer.stemWord(word))
                if len(feature) == 0:
                    continue
                if not feature.isalnum() or not feature[0].isalpha():
                    continue
                single_feature.append(feature)
                yield feature
            for i in range(1, len(single_feature)):
                yield (single_feature[i-1], single_feature[i])

    @staticmethod
    def split_word(txt):
        SPACES='[\x09\x0a\x0b\x0d\x20\xa0\'"\(\)\[\]{}\n\t\r«»,’\-–]'
        for w in re.split(SPACES,txt):
            yield w.strip()

    @staticmethod
    def split_sentances(txt):
        for s in re.split(r"[.;!:?]", txt):
            yield s

    def get_features(self):
        return set(self.__get_tf__().keys())

    def get_tf(self, vocabulary):
        vect = []
        for w in vocabulary:
            vect.append(self.__get_tf__().get(w, 0) / len(vocabulary))
        return np.float32(vect)

class Corpus:
    def __init__(self):
        self.__df__ = {}
        self.__docs__ = []

    @staticmethod
    def from_iter(iter):
        corpus = Corpus()
        for news in iter:
            corpus.__docs__.append(news)
            for t in news.get_features():
                if t not in corpus.__df__:
                    corpus.__df__[t] = 0
                corpus.__df__[t] += 1
        return corpus

    def vectorize(self, _filter=None):
        veckeys = [k for k, v in self.__df__.items() if v > 1]
        valkeys = np.float32([np.log(len(self.__docs__) / self.__df__[k]) for k in veckeys])
        #print (veckeys)
        for doc in self.iter(_filter=_filter):
            yield doc.get_tf(veckeys) * valkeys

    def __getitem__(self, i):
        return self.__docs__[i]

    def iter(self, _filter=None):
        if _filter is None:
            _filter = lambda x: True
        for doc in filter(_filter, self.__docs__):
            yield doc

class Subcorpus:
    def __init__(self, corpus, _filter):
        self.__filter__ = _filter
        self.__corpus__ = corpus

    def vectorize(self):
        return self.__corpus__.vectorize(_filter=self.__filter__)

    def __getitem__(self, i):
        items = [i for i in self.__corpus__.iter(_filter=self.__filter__)]
        return items[i]

class Analyse(Base):
    __tablename__ = "analyse"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, default=datetime.now())

    clusters = relationship(
        "Cluster", back_populates="analyse"
    )

class Cluster(Base):
    __tablename__ = "cluster"

    id = Column(Integer, primary_key=True, autoincrement=True)

    analyse_id =  Column(Integer, ForeignKey("analyse.id"))
    analyse = relationship("Analyse", back_populates="clusters")

    matches = relationship("Match")

class Match(Base):
    __tablename__ = "match"

    id = Column(Integer, primary_key=True, autoincrement=True)
    distance = Column(Numeric)

    news_id =  Column(String, ForeignKey("news.guid"))
    news = relationship("News")

    cluster_id = Column(Integer, ForeignKey("cluster.id"))
    cluster = relationship("Cluster", back_populates="matches")