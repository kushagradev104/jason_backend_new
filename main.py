import os
import json
import base64
from fastapi import FastAPI , UploadFile , File , Form , HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain_classic.agents import create_tool_calling_agent
from langchain_classic.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool

load_dotenv()
client = OpenAI

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

tools = [
    Tool(
        name = "web search",
        func = search_tool.run,
        description = "Use this tool automatically when Jason needs to find current information, live news or lookup specific online facts."
        ),
    Tool(
        name = "math_calculator",
        func = python_calculator,
        description = "Use this tool automatically when Jason needs to compute maths operations, percentages, formulas or complex values."
    )
]

llm = ChatOpenAI(model = "gpt-40-mini", temperature = 0)

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

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@app.post("/api/chat")
async def chat_endpoint(
        message: str = Form(...),
        history: str = Form(...),
        generate_voice: bool = Form(False)
):
    try:
        raw_history = json.loads(history)

        formatted_history = []
        for msg in raw_history:
            if msg.get("role") == "user":
                formatted_history.append(HumanMessage(content=msg.get("content")))
            elif msg.get("role") == "assistant":
                formatted_history.append(AItMessage(content=msg.get("content")))

        agent_response = agent_executor.invoke({
            "input": message,
            "chat_history": formatted_history
        })

        ai_reply = agent_response["output"]
        response_data = {"reply": ai_reply}

        if generate_voice:
            audio_response = client.audio.speech.create(
                model = "tts-1",
                voice = "alloy",
                input = ai_reply
            )
            base64_audio = base64.b64encode(audio_response.content).decode('utf-8')
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

        response = client.chat.completions.create(
            model = "gpt-4o-mini",
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
        return {"analysis": response.choices.message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    try:
        temp_filename = "temp_incoming_voice.wav"
        with open(temp_filename, "wb") as f:
            f.write(await audio.read())

        with open(temp_filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(model ="whisper-1", file = audio_file)

        os.remove(temp_filename)
        return {"text": transcript.text}
    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise HTTPException(status_code=500, detail=str(e))
