from flask import Flask, jsonify, request
from openai import OpenAI
import os
from flask_cors import CORS

# Initialize OpenAI client with API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app, resources={r"/generate_content": {"origins": "*"}})

@app.route('/generate_content', methods=['POST'])
def generate_content():
    if request.is_json:
        try:
            # Get the user's input from the request
            user_input = request.json.get('user_input')

            # Use OpenAI's GPT-3 to generate content based on the user's input
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
                    {"role": "user", "content": user_input}
                ]
            )

            # Extract the generated content from the response
            generated_content = completion.choices[0].message.content

            # Return the generated content as a JSON response
            return jsonify({'generated_content': generated_content})

        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid Content-Type'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
