from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
import sys

# Ensure src is in path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from CnnClassifier.utils.common import decodeImage
from CnnClassifier.pipeline.predict import PredictionPipeline


# Fix locale issues
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "input_image.jpg"   # removed double spaces
        self.classifier = PredictionPipeline(self.filename)


# Initialize client app ONCE
clApp = ClientApp()


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    return "Training done successfully!"


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        data = request.json
        if "image" not in data:
            return jsonify({"error": "No image key found"}), 400

        image = data['image']

        # Save decoded image
        decodeImage(image, clApp.filename)

        # Predict
        result = clApp.classifier.predict()
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🔥 Starting server...")
    app.run(host='0.0.0.0', port=8080)
