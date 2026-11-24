
from google import genai
from backend.config import settings

client = genai.Client(api_key=settings.google_api_key)

for model in client.models.list():
    actions = getattr(model, "supported_actions", []) or []

    if "generateContent" in actions:
        print(model.name)
