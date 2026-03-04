from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path="config/.env")
API_KEY = os.getenv("API_KEY")
print("Using API Key:", API_KEY)
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=API_KEY,
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  extra_body={},
  model="kwaipilot/kat-coder-pro:free",
  messages=[
              {
                "role": "user",
                "content": "What is the meaning of life?"
              }
            ]
)
print(completion.choices[0].message.content)