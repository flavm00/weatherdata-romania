"""`flask --app main seed` loads the county temperature series and blog posts."""
import csv
import json
from pathlib import Path

import click

from . import db
from .models import Post, TimeseriesCounty

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def seed_database():
    if not TimeseriesCounty.query.first():
        with open(DATA_DIR / "timeseries_1901-2020_ROU_county.csv", newline="", encoding="utf-8") as f:
            rows = [
                TimeseriesCounty(year=int(r["Year"]), temp=float(r["MeanTemp"]), area=r["Area"])
                for r in csv.DictReader(f)
            ]
        db.session.add_all(rows)
    if not Post.query.first():
        posts = json.loads((DATA_DIR / "posts.json").read_text(encoding="utf-8"))
        db.session.add_all(Post(**p) for p in posts)
    db.session.commit()
    return TimeseriesCounty.query.count(), Post.query.count()


def register_cli(app):
    @app.cli.command("seed")
    def seed_command():
        """Load the county temperature series and blog articles."""
        series, posts = seed_database()
        click.echo(f"Database ready: {series} temperature records, {posts} blog posts.")
