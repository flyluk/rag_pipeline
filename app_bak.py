from flask import Flask, request, render_template
import requests
import markdown

app = Flask(__name__)

VLLM_API_URL = "http://localhost:8000/v1/completions"  # Replace with your VLLM API server URL

# Home route    
@app.route("/", methods=["GET", "POST"])
# Home route function
# This function handles the GET and POST requests to the home route
def home():
    if request.method == "POST":
        input_text = request.form["input_text"]
        response = generate_response(input_text)
        response_markdown = markdown.markdown(response) 
        return render_template("index.html", response=response_markdown) 
    return render_template("index.html")
# Generate response function
def generate_response(input_text):
# This function sends a POST request to the VLLM AsPI server to generate a response
    payload = { 
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", # Replace this with your model name
        "prompt": input_text, # Replace this with your input text
        "max_tokens": 1000, # Adjust this value to control the length of the response
        "temperature": 0.5, # Adjust this value to control the randomness of the response
    }    
    
    response = requests.post(VLLM_API_URL, json=payload)
    if response.status_code == 200:
        return response.json().get("choices")[0].get("text") 
    else:
        return "Error: Unable to get response from VLLM AsPI server" # Return an error message if the request fails

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Run the Flask app on port 5000
