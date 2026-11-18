from flask_login import UserMixin
from sqlalchemy.sql import func

from . import db


class Post(db.Model):
    """Educational blog article about severe weather."""

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(10000))
    image = db.Column(db.String(250))
    description = db.Column(db.Text)
    date = db.Column(db.DateTime(timezone=True), default=func.now())


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(256))
    first_name = db.Column(db.String(150))


class Contact(db.Model):
    """Suggestions sent through the contact form."""

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150))
    name = db.Column(db.String(150))
    form_desc = db.Column(db.Text)


class TimeseriesCounty(db.Model):
    """Mean annual temperature per Romanian county (CRU dataset, 1901-2020)."""

    id = db.Column(db.Integer, primary_key=True)
    year = db.Column(db.Integer)
    temp = db.Column(db.Float(10))
    area = db.Column(db.String(100))
