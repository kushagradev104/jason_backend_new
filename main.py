from dotenv import load_dotenv

load_dotenv()

import os
import shutil
import tempfile
import traceback
import json
import base64
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import whisper
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import edge_tts
import re

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain_classic.agents import create_tool_calling_agent
from langchain_classic.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool, tool

load_dotenv()
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

app = FastAPI(title="Jason Multi Modal Reasoning Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials="True",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "online", "message": "Jason's Multimodal AI Reasoning Engine is active!"}


search_tool = TavilyAnswer(max_results=3)


def python_calculator(query: str) -> str:
    """Executes safe, precise  Python mathematical operations to avoid numerical guessing"""
    try:
        allowed_locals = {"_builtins_": None, "abs": abs, "round": round}
        return str(eval(query, {"_builtins_": None}, allowed_locals))
    except Exception as e:
        return f"Calculation error: {str(e)}.Ensure you add valid basic maths symbols"


@tool
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
        name="web_search",
        func=search_tool.run,
        description="ALWAYS use this tool when asked for real-time information, current market trends, live news, historical facts, or looking up any online information from any point in time."
    ),
    Tool(
        name="math_calculator",
        func=python_calculator,
        description="Use this tool automatically when Jason needs to compute maths operations, percentages, formulas or complex values."
    ),
    open_app_tool
]

llm = ChatOpenAI(
    model="meta-llama/llama-3.1-8b-instruct",
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    default_headers={
        "HTTP-Referer": "https://render.com",
        "X-Title": "Jason AI Agent"
    }
)


prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

@app.post("/api/chat")
async def chat_endpoint(
    message: str = Form(...),
    history: str = Form(...),
    generate_voice: bool = Form(False),
):
  try:
    raw_history = json.loads(history)
    current_time_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    dynamic_system_message = (
        f"Your name is Jason. You are a premium workspace AI assistant."
        f" Current date and time: {current_time_str}."
        " Direct instructions:"
        " 1. Do NOT output raw XML tags like <web_search> or <tool> in your final text."
        " 2. For basic common knowledge (like capitals, math, definitions), answer directly without searching."
        " 3. Reply only in clean, natural, human-readable prose."
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
        max_iterations=5,
        early_stopping_method="generate",  # <--- Forces final answer even if max iterations hit!
    )

    formatted_history = []
    for msg in raw_history:
      if msg.get("role") == "user":
        formatted_history.append(HumanMessage(content=msg.get("content")))
      elif msg.get("role") == "assistant":
        formatted_history.append(AIMessage(content=msg.get("content")))

    agent_response = await agent_executor.ainvoke(
        {"input": message, "chat_history": formatted_history}
    )

    # Extract the output string properly from the agent response dictionary
    ai_reply = agent_response.get("output", "I'm here! How can I help?")

    # Clean up any raw pseudo tool calls or XML tags produced by the LLM
    if isinstance(ai_reply, str):
        json_match = re.search(
            r"\{.*?\"search_term\"\s*:\s*\"([^\"]+)\".*?\}",
            ai_reply,
            re.DOTALL,
        )
        if json_match:
            search_query = json_match.group(1)
            try:
                search_results = tools[0].invoke(search_query)
                ai_reply = f"Based on live search results for '{search_query}': {search_results}"
            except Exception:
                ai_reply = f"Searching for '{search_query}'... Please ask your question directly."
        else:
            ai_reply = re.sub(r"</?[^>]+>", "", ai_reply).strip()
            ai_reply = ai_reply.strip('{} "')

    response_data = {"reply": ai_reply}

    if generate_voice:
        communicate = edge_tts.Communicate(ai_reply, voice="en-US-ChristopherNeural")
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]
        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        response_data["audio"] = f"data:audio/mp3;base64,{base64_audio}"

    return response_data

  except Exception as e:
      print("\n--- LOCAL CHAT ERROR ---")
      traceback.print_exc()
      print("------------------------\n")
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
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"You are Jason, analyzing this image: {prompt}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=400
        )
        return {"analysis": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

whisper_model = whisper.load_model("base")

@app.post("/api/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    temp_filename = None
    try:
        file_ext = os.path.splitext(audio.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            shutil.copyfileobj(audio.file, tmp)
            temp_filename = tmp.name

        result = whisper_model.transcribe(temp_filename)
        transcript_text = result.get("text", "").strip()

        print(f"Successfully transcribed : {transcript_text}")
        return {"text": transcript_text}

    except Exception as e:
        print("\n---LOCAL TRANSCRIBE ERROR---")
        traceback.print_exc()
        print("-----------------------------\n")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)
