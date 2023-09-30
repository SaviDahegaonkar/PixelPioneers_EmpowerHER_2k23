from flask import Flask, request, jsonify
from chatbot import generate_response_for_web

app = Flask(__name__)

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.json
    user_input = data.get('user_input', '')
    chatbot_response = generate_response_for_web(user_input)
    return jsonify({'response': chatbot_response})

if __name__ == '__main__':
    app.run(debug=True)
