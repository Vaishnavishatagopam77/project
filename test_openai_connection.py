
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("OPENAI_API_KEY not found in environment variables.")
else:
    print("OPENAI_API_KEY found in environment variables.")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        # Simple test usage
        try:
            # Using a very cheap model or just verifying client instantiation
            # To actually test, we'd need to make a call, but let's just checking client creation for now
            # and maybe list models to verify auth
            models = client.models.list()
            print("Successfully connected to OpenAI API. Models available.")
        except Exception as e:
            print(f"Error connecting to OpenAI API: {e}")
    except ImportError:
        print("openai module not installed.")
