from langchain_openai import ChatOpenAI
import os

print("Initializing ChatOpenAI...")
try:
    chat = ChatOpenAI(openai_api_key="dummy", model_name="gpt-4o")
    print("ChatOpenAI initialized.")
    # We can't actually call it without a real key, but we want to check imports and basic object
    print(f"Chat object type: {type(chat)}")
    print(f"Client type: {type(chat.client)}")
except Exception as e:
    print(f"Error: {e}")
