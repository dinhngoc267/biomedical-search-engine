from flask import Flask, render_template, request, jsonify
import biosyn.model as model
import json

app = Flask(__name__, static_folder='static')

@app.route("/", methods=['GET'])
def home():
    return render_template("index.html")

@app.route("/", methods=['POST'])
def search_query():
    data = json.loads(request.data)
    query = data.get('query')
    candidate_indices = model.model.retreival([query])
    candidate_names = model.model.dictionary.data[candidate_indices][:,0]
    return jsonify(result=1, data=candidate_names)

if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug=True, use_reloader=False) 

