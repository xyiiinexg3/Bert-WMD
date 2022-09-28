from flask import Flask,jsonify,request,render_template
from flask_jsglue import JSGlue

app = Flask(__name__)
jsglue = JSGlue(app)
app.config['DEBUG'] = True

@app.route('/')
def hello():
    return render_template("index.html")


@app.route('/index', methods=['POST','GET'])
def display(text):
    # sentence = request.form['sentence']
    res = text + "hhhhh"

    return jsonify({'sentence': res})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)