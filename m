import google.generativeai as genai

# Replace with your actual Google API key
GOOGLE_API_KEY = "AIzaSyAoAo40jhkqcl542yr-7FZ6_Oo0Pf7p8II"

# Configure the API
genai.configure(api_key=GOOGLE_API_KEY)

# List available models
def list_models():
    try:
        models = genai.list_models()
        for model in models:
            print(model.name)
    except Exception as e:
        print(f"Error: {e}")

# Call the function
list_models()
