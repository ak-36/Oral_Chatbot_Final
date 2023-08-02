from flask import Flask
from flask import render_template, request, jsonify
from langchain_chatbot import get_response, prepare_context

folder = "C:\Users\Digital Me\Desktop\Oral_Chatbot\AOMSI"
chain = prepare_context(folder)
app = Flask(__name__)


@app.get("/")
def index_get():
  return render_template("base.html")

@app.post("/predict")
def predict():
  text = request.get_json().get("message")
  response = get_response(text, chain)
  message = {"answer" : response['result']}
  print("message", message)
  print(jsonify(message))
  return jsonify(message)

if __name__ == "__main__":
  app.run(debug=True)