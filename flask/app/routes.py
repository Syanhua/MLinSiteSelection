from app import app
from flask import render_template

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/forms')
def route():
    return render_template('forms.html')
# @app.route('/index')
# def index():
#     user = {'username': 'Miguel'}
#     return render_template('index.html', title='Home', user=user)