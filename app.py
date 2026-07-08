from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# load model
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# nếu dùng Logistic thì load scaler
# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        # lấy input theo đúng thứ tự feature
        features = [
            data["proto"],
            data["sport"],
            data["dport"],
            data["pkts"],
            data["bytes"],
            data["dur"],
            data["spkts"],
            data["dpkts"],
            data["sbytes"],
            data["dbytes"],
            data["rate"],
            data["srate"],
            data["drate"],
            data["state"]
        ]

        features = np.array(features).reshape(1, -1)

        # nếu dùng Logistic:
        # features = scaler.transform(features)

        prediction = model.predict(features)[0]
        labels = {
            0: "DDoS",
            1: "Normal",
            2: "Reconaissance",
            3: "Theft"
        }

        return jsonify({
            "prediction": labels[int(prediction)]
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)