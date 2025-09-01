import requests
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import requests

model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Request schema for input
class QuestionRequest(BaseModel):
    question: str
    # model:model now would be passed from the global model.


def ask_model(question: str):
    """
    Sends a request to the model server and fetches a response.
    """
    url = "http://localhost:8000/v1/chat/completions"  # Adjust the URL if different
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()  # Raise exception for HTTP errors
    return response.json()

# Usage:
result = ask_model("What is the capital of France?")
print(json.dumps(result, indent=2))

def stream_llm_response(question:str):
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "stream": True  # 🔥 Enable streaming
    }



    with requests.post(url, headers=headers, json=data, stream=True) as response:
        for line in response.iter_lines():
            if line:
                # OpenAI-style streaming responses are prefixed with "data: "
                decoded_line = line.decode("utf-8").replace("data: ", "")
                yield decoded_line + "\n"