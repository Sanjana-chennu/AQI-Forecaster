# main.py

import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- 1. Add project directories to the Python path ---
# This ensures we can import our modules like 'bot'
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# --- 2. Import your agent creator ---
from bot.agent import create_agent

# --- 3. Initialize the FastAPI App and the Agent ---
print("Initializing FastAPI server and LangChain agent...")
app = FastAPI()
# Create the agent executor ONCE when the server starts up.
agent_executor = create_agent()
print("Agent created successfully.")


# --- 4. Add CORS Middleware ---
# This is a security feature that allows your React front-end (on a different address)
# to make requests to this back-end server.
origins = [
    "http://localhost",
    "http://localhost:3000", # The default address for a React development server
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)

# --- 5. Define the data model for the request ---
# This ensures the frontend sends data in the correct format using Pydantic.
class ChatRequest(BaseModel):
    message: str

# --- 6. Define the main API endpoint for the chat ---
@app.post("/chat")
def handle_chat(request: ChatRequest):
    """
    This endpoint receives a user's message, invokes the LangChain agent,
    and returns the agent's final text response.
    """
    print(f"Received message from frontend: {request.message}")
    
    # The core logic: run the agent with the user's input
    try:
        response = agent_executor.invoke({"input": request.message})
        # Extract the final text answer from the agent's output
        bot_response = response.get('output', 'Sorry, I had trouble processing that request.')
    except Exception as e:
        print(f"An error occurred in the agent: {e}")
        bot_response = "Sorry, an internal error occurred."
    
    print(f"Sending response to frontend: {bot_response}")
    return {"reply": bot_response}

print("FastAPI server setup complete. Run 'uvicorn main:app --reload' to start.")