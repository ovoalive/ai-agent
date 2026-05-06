import os
import uuid
from datetime import datetime
from typing import Dict, Optional, List, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 4096
    openai_temperature: float = 0.7

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()

try:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.openai_api_key if settings.openai_api_key else os.getenv("OPENAI_API_KEY"))
except ImportError:
    client = None


class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    conversation_id: str
    message: str
    total_tokens: Optional[int] = None
    model: Optional[str] = None


class Conversation(BaseModel):
    conversation_id: str
    messages: List[Message]
    created_at: datetime
    updated_at: datetime


class ConversationStore:
    def __init__(self):
        self.conversations: Dict[str, Conversation] = {}

    def create_conversation(self) -> Conversation:
        conversation_id = str(uuid.uuid4())
        now = datetime.now()
        conversation = Conversation(
            conversation_id=conversation_id,
            messages=[],
            created_at=now,
            updated_at=now
        )
        self.conversations[conversation_id] = conversation
        return conversation

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        return self.conversations.get(conversation_id)

    def add_message(self, conversation_id: str, role: str, content: str) -> Optional[Conversation]:
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return None
        message = Message(role=role, content=content, timestamp=datetime.now())
        conversation.messages.append(message)
        conversation.updated_at = datetime.now()
        return conversation

    def delete_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return True
        return False

    def list_conversations(self) -> List[Conversation]:
        return sorted(self.conversations.values(), key=lambda c: c.updated_at, reverse=True)

    def clear_all(self):
        self.conversations.clear()


conversation_store = ConversationStore()

app = FastAPI(title="LLM Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def chat_completion(messages: List[Dict[str, str]], model=None, max_tokens=None, temperature=None):
    model = model or settings.openai_model
    max_tokens = max_tokens or settings.openai_max_tokens
    temperature = temperature or settings.openai_temperature
    
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False
    )
    
    return {
        "content": response.choices[0].message.content,
        "total_tokens": response.usage.total_tokens,
        "model": response.model
    }


async def stream_chat_completion(messages: List[Dict[str, str]], model=None, max_tokens=None, temperature=None):
    model = model or settings.openai_model
    max_tokens = max_tokens or settings.openai_max_tokens
    temperature = temperature or settings.openai_temperature
    
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    
    if not request.conversation_id:
        conversation = conversation_store.create_conversation()
        conversation_id = conversation.conversation_id
    else:
        conversation = conversation_store.get_conversation(request.conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conversation_id = request.conversation_id

    conversation_store.add_message(conversation_id, "user", request.message)
    messages = [{"role": m.role, "content": m.content} for m in conversation.messages]
    result = await chat_completion(messages)
    conversation_store.add_message(conversation_id, "assistant", result["content"])

    return ChatResponse(
        conversation_id=conversation_id,
        message=result["content"],
        total_tokens=result["total_tokens"],
        model=result["model"]
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    
    if not request.conversation_id:
        conversation = conversation_store.create_conversation()
        conversation_id = conversation.conversation_id
    else:
        conversation = conversation_store.get_conversation(request.conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conversation_id = request.conversation_id

    conversation_store.add_message(conversation_id, "user", request.message)
    messages = [{"role": m.role, "content": m.content} for m in conversation.messages]

    async def generate():
        full_response = ""
        async for chunk in stream_chat_completion(messages):
            full_response += chunk
            yield chunk
        conversation_store.add_message(conversation_id, "assistant", full_response)

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/conversations")
async def list_conversations():
    return {"conversations": conversation_store.list_conversations()}


@app.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    conversation = conversation_store.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    success = conversation_store.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation deleted successfully"}


@app.delete("/conversations")
async def clear_conversations():
    conversation_store.clear_all()
    return {"message": "All conversations cleared successfully"}


HTML_CONTENT = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Assistant - AI智能助手</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            width: 100%; max-width: 800px; background: white; border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3); overflow: hidden;
            display: flex; flex-direction: column; height: 90vh;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px; color: white; text-align: center;
        }
        .header h1 { font-size: 24px; font-weight: 600; }
        .header p { font-size: 14px; opacity: 0.9; margin-top: 5px; }
        .chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; }
        .message {
            max-width: 70%; margin-bottom: 15px; padding: 12px 18px;
            border-radius: 20px; line-height: 1.5; animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .user-message {
            align-self: flex-end;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border-bottom-right-radius: 5px;
        }
        .assistant-message {
            align-self: flex-start; background: #f1f1f1; color: #333;
            border-bottom-left-radius: 5px;
        }
        .message-timestamp { font-size: 11px; opacity: 0.6; margin-top: 5px; text-align: right; }
        .input-container { padding: 20px; border-top: 1px solid #eee; display: flex; gap: 10px; }
        .input-container input {
            flex: 1; padding: 15px 20px; border: 2px solid #eee; border-radius: 30px;
            font-size: 16px; transition: border-color 0.3s; outline: none;
        }
        .input-container input:focus { border-color: #667eea; }
        .input-container button {
            padding: 15px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border: none; border-radius: 30px; font-size: 16px;
            cursor: pointer; transition: transform 0.2s, box-shadow 0.2s;
        }
        .input-container button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102,126,234,0.4); }
        .typing-indicator { display: flex; gap: 4px; padding: 12px 18px; }
        .typing-dot {
            width: 8px; height: 8px; background: #ccc; border-radius: 50%;
            animation: typing 1.4s infinite ease-in-out both;
        }
        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }
        .typing-dot:nth-child(3) { animation-delay: 0s; }
        @keyframes typing { 0%, 80%, 100% { transform: scale(0.6); opacity: 0.5; } 40% { transform: scale(1); opacity: 1; } }
        .empty-state { flex: 1; display: flex; flex-direction: column; justify-content: center; align-items: center; color: #999; }
        .empty-state svg { width: 64px; height: 64px; margin-bottom: 20px; opacity: 0.5; }
        .empty-state h3 { font-size: 18px; margin-bottom: 10px; color: #666; }
        .empty-state p { font-size: 14px; }
        @media (max-width: 480px) { .container { height: 100vh; border-radius: 0; } .message { max-width: 85%; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 LLM Assistant</h1>
            <p>基于大语言模型的智能对话助手</p>
        </div>
        <div class="chat-container" id="chatContainer">
            <div class="empty-state" id="emptyState">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                </svg>
                <h3>开始对话</h3>
                <p>输入您的问题，我将为您提供帮助</p>
            </div>
        </div>
        <div class="input-container">
            <input type="text" id="messageInput" placeholder="输入消息..." autocomplete="off"
                onkeydown="if(event.keyCode===13) sendMessage()">
            <button id="sendButton" onclick="sendMessage()">
                <span id="sendText">发送</span>
                <span id="loadingIndicator" style="display:none;">⏳</span>
            </button>
        </div>
    </div>
    <script>
        const chatContainer = document.getElementById('chatContainer');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const sendText = document.getElementById('sendText');
        const loadingIndicator = document.getElementById('loadingIndicator');
        const emptyState = document.getElementById('emptyState');
        
        let currentConversationId = null;
        let isTyping = false;

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message || isTyping) return;

            isTyping = true;
            messageInput.value = '';
            sendText.style.display = 'none';
            loadingIndicator.style.display = 'inline';

            if (emptyState) emptyState.remove();
            addMessage(message, 'user');

            const typingIndicator = document.createElement('div');
            typingIndicator.className = 'assistant-message typing-indicator';
            typingIndicator.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
            chatContainer.appendChild(typingIndicator);
            chatContainer.scrollTop = chatContainer.scrollHeight;

            try {
                const response = await fetch('/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: message,
                        conversation_id: currentConversationId || undefined
                    })
                });

                if (!response.ok) throw new Error('请求失败');

                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let fullResponse = '';

                typingIndicator.remove();
                const assistantMessage = document.createElement('div');
                assistantMessage.className = 'assistant-message';
                chatContainer.appendChild(assistantMessage);

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    fullResponse += decoder.decode(value, { stream: true });
                    assistantMessage.textContent = fullResponse;
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }

                const finalResponse = await fetch('/conversations');
                const data = await finalResponse.json();
                if (data.conversations.length > 0) {
                    currentConversationId = data.conversations[0].conversation_id;
                }
            } catch (error) {
                typingIndicator?.remove();
                addMessage('抱歉，服务暂时不可用，请检查API密钥配置。', 'assistant');
            } finally {
                isTyping = false;
                sendText.style.display = 'inline';
                loadingIndicator.style.display = 'none';
            }
        }

        function addMessage(content, role) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `${role}-message message`;
            messageDiv.textContent = content;
            
            const timestamp = document.createElement('div');
            timestamp.className = 'message-timestamp';
            const now = new Date();
            timestamp.textContent = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
            messageDiv.appendChild(timestamp);
            
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_CONTENT


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)