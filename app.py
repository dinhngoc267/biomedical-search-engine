from flask import Flask, render_template, request, jsonify
import biosyn.model as model
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

json_dump = json.dumps({'a': a, 'aa': [2, (2, 3, 4), a], 'bb': [2]}, cls=NumpyEncoder)
print(json_dump)

app = Flask(__name__, static_folder='static')

@app.route("/", methods=['GET'])
def main():
    print('here')
    return render_template("index.html")

@app.route("/", methods=['POST'])
def search_query():
    data = json.loads(request.data)
    query = data.get('query')
    candidate_indices = model.model.retreival([query])
    candidate_names = model.model.dictionary.data[candidate_indices]
    candidate_names = candidate_names.squeeze()[:,0]
    #json_dump = json.dumps({'data': candidate_names}, cls=NumpyEncoder)
    return jsonify(result=1, data=candidate_names)

if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug=True, use_reloader=False, host='0.0.0.0',  port=8080) 

