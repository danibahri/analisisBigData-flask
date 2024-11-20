from flask import Flask, request, jsonify, render_template, url_for


app = Flask(__name__)

@app.route('/', methods=['GET'])
@app.route('/home/', methods=['GET'])
def index():
    return render_template('index.html')

    

app.run(debug=True)