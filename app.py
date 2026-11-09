import os
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI

# Load environment variables from a .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes (adjust origins in production if needed)

# Initialize the OpenAI client configured for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# System prompt defining the chatbot's premium, warm, and natural personality
SYSTEM_PROMPT = (
    "You are a premium, sophisticated personal assistant. "
    "Your tone is warm, genuinely helpful, empathetic, and entirely natural—never robotic or overly formal. "
    "Always provide well-structured, clear, and easy-to-read responses using proper Markdown formatting when appropriate. "
    "Treat each interaction as a seamless, ongoing conversation, making the user feel supported, understood, and valued."
)


@app.route("/chat", methods=["POST"])
def chat():
  try:
    data = request.get_json()

    # Validate incoming JSON payload
    if not data or "message" not in data:
      return (
          jsonify({"error": "Invalid request: 'message' field is required."}),
          400,
      )

    user_message = data.get("message")
    history = data.get("history", [])

    # Build the conversation message chain
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Append conversation history if provided
    for msg in history:
      role = msg.get("role")
      content = msg.get("content")
      if role in ["user", "assistant"] and content:
        messages.append({"role": role, "content": content})

    # Append the latest user message
    messages.append({"role": "user", "content": user_message})

    # Call OpenRouter API using the OpenAI SDK
    response = client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=messages,
        temperature=0.7,
    )

    bot_reply = response.choices[0].message.content

    return jsonify({"response": bot_reply}), 200

  except Exception as e:
    # Log error internally in production and return a clean error message
    return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
  port = int(os.getenv("PORT", 5000))
  app.run(host="0.0.0.0", port=port, debug=False)
