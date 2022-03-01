from flask import Flask, render_template, request

app = Flask(__name__)


# @ is a decorator - a way to wrap a function and modify its behavior
@app.route('/')
def index():
    return render_template("notebook.html")


if __name__ == '__main__':
    app.run()
# Changes made
