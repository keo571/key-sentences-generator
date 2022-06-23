from flask import Flask, render_template, request, flash
from ksg import key_sen_gen

app = Flask(__name__)
app.config['SECRET_KEY'] = 'd341fa5c1b56b3b96c28e4b585263325b92b66d8678a4b46'


@app.route('/', methods=('GET', 'POST'))
def index():
    sentences = []

    if request.method == 'POST':
        content = request.form['content']

        if not content:
            flash('Content is required!')
        else:
            words, sentences = key_sen_gen(content)

    return render_template('index.html', sentences=sentences)


@app.route('/about/')
def about():
    return render_template('about.html')
