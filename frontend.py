from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from models import NewsStream, News, Analyse, Cluster, Match

db = SQLAlchemy()
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////home/droper/news.db"
db.init_app(app)

def cluster_postproc(cluster):
    cluster.matches.sort(key=lambda x: x.distance)
    title, description, media = None, None, None
    for match in cluster.matches:
        if title is None:
            title = match.news.title
        if description is None or description == title:
            description = match.news.description
        if media is None:
            media = match.news.media
    return {
        "id": cluster.id,
        "title": title,
        "description": description,
        "media": media,
        "sources": [match.news for match in cluster.matches]
    }

@app.route("/")
def index():
    analyses = db.session.execute(db.select(Analyse).order_by(Analyse.id)).scalars()
    return render_template("index.html", analyses=analyses)

@app.route("/analyse/<int:id>")
def analyse(id):
    analyse = db.get_or_404(Analyse, id)
    clusters = [cluster_postproc(cluster) for cluster in analyse.clusters]
    return render_template("analyse.html", clusters=clusters, date=analyse.date)

@app.route("/cluster/<int:id>")
def cluster(id):
    cluster = db.get_or_404(Cluster, id)
    data = cluster_postproc(cluster)
    return render_template("cluster.html", data=data)