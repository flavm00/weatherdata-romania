from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from . import db
from .models import Contact, Post, TimeseriesCounty

views = Blueprint("views", __name__)


@views.route("/", methods=["GET"])
def home():
    return render_template("home.html", user=current_user)


@views.route("/blog", methods=["GET"])
@login_required
def blog():
    posts = Post.query.all()
    return render_template("blog.html", user=current_user, posts=posts)


@views.route("/blog/<int:post_id>/")
@login_required
def display_article(post_id):
    post = db.get_or_404(Post, post_id)
    return render_template("post.html", user=current_user, post=post)


@views.route("/data_services", methods=["GET"])
@login_required
def data_services():
    county = TimeseriesCounty.query.group_by(TimeseriesCounty.area).all()
    data = TimeseriesCounty.query.all()
    return render_template("data_services.html", user=current_user, county=county, data=data)


@views.route("/contact", methods=["GET", "POST"])
@login_required
def contact():
    if request.method == "POST":
        name = request.form.get("name", "")
        form_desc = request.form.get("form_desc", "")
        if len(name) < 5:
            flash("Numele trebuie să fie complet", category="error")
        else:
            db.session.add(Contact(email=current_user.email, name=name, form_desc=form_desc))
            db.session.commit()
            flash("Sugestia a fost trimisă cu succes!", category="success")
            return redirect(url_for("views.contact"))
    return render_template("contact.html", user=current_user)
