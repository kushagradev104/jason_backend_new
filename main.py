from dotenv import load_dotenv
load_dotenv()

import os
import json
import base64
from datetime import datetime
from fastapi import FastAPI , UploadFile , File , Form , HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import edge_tts


from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain_classic.agents import create_tool_calling_agent
from langchain_classic.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool

load_dotenv()
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)


app = FastAPI(title="Jason Multi Modal Reasoning Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials = "True",
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return{"status": "online", "message": "Jason's Multimodal AI Reasoning Engine is active!"}

search_tool = TavilyAnswer(max_results=3)

def python_calculator(query: str) -> str:
    """Executes safe, precise  Python mathematical operations to avoid numerical guessing"""
    try:
        allowed_locals = {"_builtins_":None, "abs": abs, "round": round}
        return str(eval(query,{"_builtins_": None },allowed_locals))
    except Exception as e:
        return f"Calculation error: {str(e)}.Ensure you add valid basic maths symbols"

@Tool
def open_app_tool(app_name: str) -> str:
    """
    Use this tool when the user explicitly requests to open a desktop app or software 
    (e.g., Spotify, Discord, VS Code, YouTube, Calculator, Chrome).
    """
    app_clean = app_name.lower().strip()
    
    # Return a structured JSON payload so the client/frontend executes it natively
    payload = {
        "status": "success",
        "action": "OPEN_APP",
        "app": app_clean,
        "message": f"Opening {app_name} for you now!"
    }
    
    return json.dumps(payload)
    SYSTEM_PROMPT = """
You are Jason, a fast and concise AI workspace assistant.

STRICT TOOL INSTRUCTIONS:
1. When the user asks to open an app (e.g., "open Spotify"), call the `open_app_tool`.
2. Do NOT invent URLs or say you cannot open local files.
3. Keep your spoken response brief and natural for Edge-TTS audio (e.g., "Opening Spotify!").
"""

tools = [
    Tool(
        name = "web_search",
        func = search_tool.run,
        description = "ALWAYS use this tool when asked for real-time information, current market trends, live news, historical facts, or looking up any online information from any point in time."
    ),
    Tool(
        name = "math_calculator",
        func = python_calculator,
        description = "Use this tool automatically when Jason needs to compute maths operations, percentages, formulas or complex values."
    ),
    Tool(
        name = "open_app_tool",
        func = open_app_tool, # Or just open_app_tool depending on how it's invoked
        description = "Use this tool when the user explicitly requests to open a desktop app or software like Spotify, Discord, VS Code, or Calculator."
    )
]


llm = ChatOpenAI(
    model = "meta-llama/llama-3.1-8b-instruct", 
    temperature = 0,
    base_url = "https://openrouter.ai/api/v1",
    api_key = os.getenv("OPENAI_API_KEY"),
    default_headers={
        "HTTP-Referer": "https://render.com",
        "X-Title": "Jason AI Agent"
    }
)

system_message = (
    "Your name is Jason. You are a premium, highly competent workspace AI assistant. "
    "Always introduce yourself as Jason when greeted for the first time. "
    "You have automated tools for web searching and high-precision calculations. Use them contextually "
    "whenever the user requests real-time data or mathematical solutions. "
    "Keep your replies conversational , accurate and structured. "
)

prompt = ChatPromptTemplate.from_messages([
    ("system",system_message),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

@app.post("/api/chat")
async def chat_endpoint(
        message: str = Form(...),
        history: str = Form(...),
        generate_voice: bool = Form(False)
):
    try:
        raw_history = json.loads(history)
        current_time_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

        dynamic_system_message = (
            f"Your name is Jason. You are a premium, highly competent workspace AI assistant. "
            f"The current date and time is {current_time_str}. "
            f"Always introduce yourself as Jason when greeted for the first time. "
            f"You have automated tools for web searching and high-precision calculations. You MUST use them contextually "
            f"whenever the user requests real-time data, current events, or mathematical solutions. "
            f"Never mention knowledge cutoffs, training data limits, or state that you cannot browse the internet. "
            f"Keep your replies conversational, accurate, and structured."
        )

        current_prompt = ChatPromptTemplate.from_messages([
        ("system", dynamic_system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

        current_agent = create_tool_calling_agent(llm, tools, current_prompt)
        agent_executor = AgentExecutor(
            agent=current_agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )


        formatted_history = []
        for msg in raw_history:
            if msg.get("role") == "user":
                formatted_history.append(HumanMessage(content=msg.get("content")))
            elif msg.get("role") == "assistant":
                formatted_history.append(AIMessage(content=msg.get("content")))

        agent_response = await agent_executor.ainvoke({
            "input": message,
            "chat_history": formatted_history
        })
        
        ai_reply = agent_response["output"]
        response_data = {"reply": ai_reply}
        
        if generate_voice:
            communicate = edge_tts.Communicate(ai_reply, "en-US-ChristopherNeural")
            audio_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]
            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
            response_data["audio"] = f"data:audio/mp3;base64,{base64_audio}"

        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vision")
async def vision_endpoint(
        image: UploadFile = File(...),
        prompt: str = Form("Analyze this photo taken by the user")
):

    try:
        file_bytes = await image.read()
        base64_image = base64.b64encode(file_bytes).decode('utf-8')

        response = openrouter_client.chat.completions.create(
            model = "openai/gpt-4o-mini",
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"You are Jason, analyzing this image: {prompt}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens = 400
        )
        return {"analysis": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    try:
        temp_filename = "temp_incoming_voice.wav"
        with open(temp_filename, "wb") as f:
            f.write(await audio.read())

        with open(temp_filename, "rb") as audio_file:
            transcript = openrouter_client.audio.transcriptions.create(model ="whisper-1", file = audio_file)

        os.remove(temp_filename)
        return {"text": transcript.text}
    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise HTTPException(status_code=500, detail=str(e))

