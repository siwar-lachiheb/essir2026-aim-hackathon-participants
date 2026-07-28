import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
    
print("BASE_URL:", os.getenv("LITELLM_BASE_URL"))
print("API_KEY:", os.getenv("LITELLM_API_KEY"))
print("MODEL:", os.getenv("CHAT_MODEL"))


client = OpenAI(
    api_key=os.getenv("LITELLM_API_KEY"),
    base_url=os.getenv("LITELLM_API_BASE"),
)

response = client.chat.completions.create(
    model=os.getenv("CHAT_MODEL"),
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message.content)