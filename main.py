import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
from twilio.rest import Client
from dotenv import load_dotenv
import requests

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
PORT = int(os.getenv('PORT', 5050))
SYSTEM_MESSAGE = (
    "You are an AI assistant for AryaMind Technologies. You should ONLY provide information about "
    "AryaMind Technologies, including their services, expertise, and contact information. "
    "For any other topics, respond with 'I apologize, but I don't have information about that. "
    "I can only provide information about AryaMind Technologies. Would you like to ask something about AryaMind, "
    "or should I end our call?' Always maintain a professional and helpful tone."
)
VOICE = 'alloy'
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created'
]
SHOW_TIMING_MATH = False
NGROK_URL = os.getenv('NGROK_URL')
RASA_URL = "http://localhost:5005/webhooks/rest/webhook"

app = FastAPI()

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')
if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
    raise ValueError('Missing Twilio credentials. Please set them in the .env file.')
if not NGROK_URL:
    raise HTTPException(status_code=500, detail="NGROK_URL is not set in the environment variables")

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}

@app.post("/make-call")
async def make_call(request: Request):
    """Initiate an outbound call to a specified phone number."""
    try:
        data = await request.json()
        to_number = data.get('to')
        
        if not to_number:
            raise HTTPException(status_code=400, detail="Phone number is required")
        
        # Create a call using Twilio
        print(f"Twilio call webhook URL: {NGROK_URL}/outbound-call")
        call = twilio_client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=f'{NGROK_URL}/outbound-call'
        )
        
        return {"message": "Call initiated", "call_sid": call.sid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.api_route("/outbound-call", methods=["GET", "POST"])
async def handle_outbound_call(request: Request):
    """Handle outbound call and return TwiML response to connect to Media Stream."""
    try:
        response = VoiceResponse()
        # <Say> punctuation to improve text-to-speech flow
        response.say("Please wait while we connect your call to the AI voice assistant, powered by AryaMind Technologies")
        response.pause(length=1)
        response.say("O.K. you can start talking!")
        
        # Get the full URL including protocol
        if not NGROK_URL:
            raise HTTPException(status_code=500, detail="NGROK_URL is not set in environment variables")
            
        # Remove any trailing slashes from NGROK_URL
        base_url = NGROK_URL.rstrip('/')
        ws_url = f"{base_url}/media-stream"
        
        print(f"WebSocket URL: {ws_url}")
        
        connect = Connect()
        connect.stream(url=ws_url)
        response.append(connect)
        
        return HTMLResponse(content=str(response), media_type="application/xml")
    except Exception as e:
        print(f"Error in handle_outbound_call: {str(e)}")
        response = VoiceResponse()
        response.say("We're sorry, but we're experiencing technical difficulties. Please try again later.")
        return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI."""
    try:
        print("Client attempting to connect...")
        await websocket.accept()
        print("Client connected successfully")

        if not OPENAI_API_KEY:
            print("Error: OPENAI_API_KEY is not set")
            await websocket.close(code=1008, reason="Server configuration error")
            return

        try:
            async with websockets.connect(
                'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
                extra_headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "OpenAI-Beta": "realtime=v1"
                }
            ) as openai_ws:
                print("Connected to OpenAI WebSocket")
                await initialize_session(openai_ws)

                # Connection specific state
                stream_sid = None
                latest_media_timestamp = 0
                last_assistant_item = None
                mark_queue = []
                response_start_timestamp_twilio = None

                try:
                    await asyncio.gather(
                        receive_from_twilio(),
                        send_to_twilio()
                    )
                except Exception as e:
                    print(f"Error in WebSocket communication: {str(e)}")
                    await websocket.close(code=1011, reason="Internal server error")

        except Exception as e:
            print(f"Error connecting to OpenAI: {str(e)}")
            await websocket.close(code=1011, reason="OpenAI connection error")

    except WebSocketDisconnect:
        print("Client disconnected normally")
    except Exception as e:
        print(f"Unexpected error in WebSocket handler: {str(e)}")
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass

async def send_initial_conversation_item(openai_ws):
    """Send initial conversation item if AI talks first."""
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Hello! I am Aryabot, your AI assistant from AryaMind Technologies. "
                        "Ask me anything about our services and expertise!"
                    )
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))


async def initialize_session(openai_ws):
    """Control initial session with OpenAI."""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

    # Uncomment the next line to have the AI speak first
    # await send_initial_conversation_item(openai_ws)

def get_rasa_response(text, sender="user"):
    print(f"Sending to Rasa: {text}")
    payload = {"sender": sender, "message": text}
    try:
        response = requests.post(RASA_URL, json=payload)
        response.raise_for_status()
        rasa_replies = response.json()
        print(f"Rasa replied: {rasa_replies}")
        if rasa_replies:
            return rasa_replies[0].get('text', '')
        return ''
    except Exception as e:
        print(f"Error communicating with Rasa: {e}")
        return ''

def is_rasa_fallback(rasa_response):
    fallback_texts = [
        "I didn't understand that.",
        "Sorry, I didn't get that.",
        "default fallback intent"
    ]
    return rasa_response.strip().lower() in [t.lower() for t in fallback_texts]

async def get_final_response(user_text):
    """Get the final response combining Rasa and OpenAI responses."""
    # First try Rasa
    rasa_response = get_rasa_response(user_text)
    
    # Check if it's a fallback response
    if is_rasa_fallback(rasa_response):
        # If Rasa falls back, use OpenAI
        openai_response = await get_openai_response(user_text)
        return openai_response
    else:
        return rasa_response

async def get_openai_response(user_text):
    """Get response from OpenAI API."""
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_text}
                ],
                "temperature": 0.7,
                "max_tokens": 150
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error getting OpenAI response: {e}")
        return "I apologize, but I'm having trouble processing your request right now. Would you like to try again or end our call?"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
