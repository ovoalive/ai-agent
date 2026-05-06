import os
import uuid
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List, Any, Callable

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 4096
    openai_temperature: float = 0.7
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-sonnet-20240229"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_conversations: int = 100
    max_messages_per_conversation: int = 50
    enable_tool_calling: bool = True
    enable_multi_agent: bool = True

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()

try:
    from openai import AsyncOpenAI
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key if settings.openai_api_key else os.getenv("OPENAI_API_KEY"))
except ImportError:
    openai_client = None

try:
    from anthropic import AsyncAnthropic
    anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key if settings.anthropic_api_key else os.getenv("ANTHROPIC_API_KEY"))
except ImportError:
    anthropic_client = None


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Dict[str, str]


class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    agent_name: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    model: Optional[str] = "openai"
    system_prompt: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096
    agent_role: Optional[str] = "general"


class ChatResponse(BaseModel):
    conversation_id: str
    message: str
    total_tokens: Optional[int] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    agent_name: Optional[str] = None


class Conversation(BaseModel):
    conversation_id: str
    messages: List[Message]
    created_at: datetime
    updated_at: datetime
    system_prompt: Optional[str] = None
    model: str = "openai"
    agent_role: str = "general"


class ModelInfo(BaseModel):
    name: str
    provider: str
    available: bool
    max_tokens: int


class SystemStats(BaseModel):
    total_conversations: int
    total_messages: int
    active_conversations: int
    total_tool_calls: int


class AgentInfo(BaseModel):
    name: str
    role: str
    description: str
    enabled: bool


class ConversationStore:
    def __init__(self, max_conversations: int = 100):
        self.conversations: Dict[str, Conversation] = {}
        self.max_conversations = max_conversations
        self.total_tool_calls = 0

    def create_conversation(self, model: str = "openai", system_prompt: Optional[str] = None, agent_role: str = "general") -> Conversation:
        if len(self.conversations) >= self.max_conversations:
            oldest_id = min(self.conversations.keys(), key=lambda k: self.conversations[k].created_at)
            del self.conversations[oldest_id]
        
        conversation_id = str(uuid.uuid4())
        now = datetime.now()
        conversation = Conversation(
            conversation_id=conversation_id,
            messages=[],
            created_at=now,
            updated_at=now,
            system_prompt=system_prompt,
            model=model,
            agent_role=agent_role
        )
        self.conversations[conversation_id] = conversation
        logger.info(f"Created new conversation: {conversation_id[:8]}")
        return conversation

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        return self.conversations.get(conversation_id)

    def add_message(self, conversation_id: str, role: str, content: str, 
                    tool_calls: Optional[List[ToolCall]] = None, tool_call_id: Optional[str] = None,
                    agent_name: Optional[str] = None) -> Optional[Conversation]:
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return None
        
        if len(conversation.messages) >= settings.max_messages_per_conversation:
            conversation.messages = conversation.messages[-settings.max_messages_per_conversation + 1:]
        
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            agent_name=agent_name
        )
        conversation.messages.append(message)
        conversation.updated_at = datetime.now()
        return conversation

    def increment_tool_calls(self):
        self.total_tool_calls += 1

    def delete_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Deleted conversation: {conversation_id[:8]}")
            return True
        return False

    def list_conversations(self) -> List[Conversation]:
        return sorted(self.conversations.values(), key=lambda c: c.updated_at, reverse=True)

    def clear_all(self):
        count = len(self.conversations)
        self.conversations.clear()
        logger.info(f"Cleared {count} conversations")

    def get_stats(self) -> SystemStats:
        total_messages = sum(len(c.messages) for c in self.conversations.values())
        return SystemStats(
            total_conversations=len(self.conversations),
            total_messages=total_messages,
            active_conversations=len(self.conversations),
            total_tool_calls=self.total_tool_calls
        )


conversation_store = ConversationStore(max_conversations=settings.max_conversations)

AGENTS = {
    "general": AgentInfo(
        name="通用助手",
        role="general",
        description="具备多种能力的通用AI助手，能够处理各种日常任务和问题",
        enabled=True
    ),
    "analyst": AgentInfo(
        name="数据分析专家",
        role="analyst",
        description="擅长数据分析、统计计算和图表解读，提供专业的数据洞察",
        enabled=True
    ),
    "writer": AgentInfo(
        name="文案创作专家",
        role="writer",
        description="专业文案撰写助手，擅长营销文案、报告撰写和创意写作",
        enabled=True
    ),
    "code": AgentInfo(
        name="代码助手",
        role="code",
        description="精通多种编程语言，提供代码编写、调试和优化建议",
        enabled=True
    ),
    "translator": AgentInfo(
        name="翻译专家",
        role="translator",
        description="多语言翻译助手，支持文档翻译和本地化服务",
        enabled=True
    )
}

AGENT_SYSTEM_PROMPTS = {
    "general": "你是一个乐于助人的AI助手，能够回答各种问题并提供有用的信息。",
    "analyst": "你是一位专业的数据分析师，擅长处理和解读数据。请使用结构化的方式呈现分析结果，包括关键发现、趋势分析和建议。",
    "writer": "你是一位资深文案策划师，擅长创作高质量的营销文案、报告和创意内容。请确保语言优美、逻辑清晰。",
    "code": "你是一位高级软件工程师，精通Python、JavaScript、Java等多种编程语言。请提供清晰、高效、可维护的代码。",
    "translator": "你是一位专业翻译专家，精通中英互译。请提供准确、自然、符合目标语言习惯的翻译结果。"
}


def get_current_time() -> str:
    return f"当前时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}"


def calculate_square_root(number: float) -> str:
    if number < 0:
        return "错误：无法计算负数的平方根"
    return f"{number} 的平方根是 {number ** 0.5:.4f}"


def calculate_complex_expression(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": None}, {"abs": abs, "pow": pow, "sqrt": lambda x: x**0.5})
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


def extract_keywords(text: str) -> str:
    keywords_prompt = f"请从以下文本中提取最重要的5-10个关键词：\n\n{text}\n\n关键词（用逗号分隔）："
    return f"关键词提取任务已触发，正在分析文本：{text[:50]}..."


AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前时间和日期",
            "parameters": {}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_square_root",
            "description": "计算一个数的平方根",
            "parameters": {
                "number": {"type": "number", "description": "要计算平方根的数字"}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_complex_expression",
            "description": "计算数学表达式",
            "parameters": {
                "expression": {"type": "string", "description": "数学表达式，如：2+3*4"}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_keywords",
            "description": "从文本中提取关键词",
            "parameters": {
                "text": {"type": "string", "description": "要分析的文本"}
            }
        }
    }
]

TOOL_FUNCTIONS: Dict[str, Callable] = {
    "get_current_time": get_current_time,
    "calculate_square_root": calculate_square_root,
    "calculate_complex_expression": calculate_complex_expression,
    "extract_keywords": extract_keywords
}


async def call_openai(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo", 
                     temperature: float = 0.7, max_tokens: int = 4096,
                     system_prompt: Optional[str] = None, tools: Optional[List] = None):
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    
    system_messages = []
    if system_prompt:
        system_messages.append({"role": "system", "content": system_prompt})
    
    all_messages = system_messages + messages
    
    response = await openai_client.chat.completions.create(
        model=model,
        messages=all_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice="auto" if tools else None
    )
    
    return {
        "content": response.choices[0].message.content,
        "total_tokens": response.usage.total_tokens,
        "model": response.model,
        "finish_reason": response.choices[0].finish_reason,
        "tool_calls": response.choices[0].message.tool_calls
    }


async def call_anthropic(messages: List[Dict[str, str]], model: str = "claude-3-sonnet-20240229",
                        temperature: float = 0.7, max_tokens: int = 4096,
                        system_prompt: Optional[str] = None):
    if not anthropic_client:
        raise HTTPException(status_code=500, detail="Anthropic client not initialized")
    
    formatted_messages = []
    for msg in messages:
        if msg["role"] == "user":
            formatted_messages.append({"role": "user", "content": [{"type": "text", "text": msg["content"]}]})
        else:
            formatted_messages.append({"role": "assistant", "content": [{"type": "text", "text": msg["content"]}]})
    
    response = await anthropic_client.messages.create(
        model=model,
        messages=formatted_messages,
        system=system_prompt or "",
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return {
        "content": response.content[0].text if response.content else "",
        "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        "model": response.model,
        "finish_reason": response.stop_reason
    }


async def stream_openai(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo",
                       temperature: float = 0.7, max_tokens: int = 4096,
                       system_prompt: Optional[str] = None):
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    
    system_messages = []
    if system_prompt:
        system_messages.append({"role": "system", "content": system_prompt})
    
    all_messages = system_messages + messages
    
    stream = await openai_client.chat.completions.create(
        model=model,
        messages=all_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


async def stream_anthropic(messages: List[Dict[str, str]], model: str = "claude-3-sonnet-20240229",
                          temperature: float = 0.7, max_tokens: int = 4096,
                          system_prompt: Optional[str] = None):
    if not anthropic_client:
        raise HTTPException(status_code=500, detail="Anthropic client not initialized")
    
    formatted_messages = []
    for msg in messages:
        if msg["role"] == "user":
            formatted_messages.append({"role": "user", "content": [{"type": "text", "text": msg["content"]}]})
        else:
            formatted_messages.append({"role": "assistant", "content": [{"type": "text", "text": msg["content"]}]})
    
    async with anthropic_client.messages.stream(
        model=model,
        messages=formatted_messages,
        system=system_prompt or "",
        temperature=temperature,
        max_tokens=max_tokens
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def multi_agent_orchestrator(user_query: str, agent_role: str = "general") -> dict:
    if agent_role == "general":
        if "分析" in user_query or "数据" in user_query or "统计" in user_query:
            agent_role = "analyst"
        elif "写" in user_query or "文案" in user_query or "报告" in user_query:
            agent_role = "writer"
        elif "代码" in user_query or "编程" in user_query or "开发" in user_query:
            agent_role = "code"
        elif "翻译" in user_query or "英文" in user_query or "English" in user_query:
            agent_role = "translator"
    
    system_prompt = AGENT_SYSTEM_PROMPTS.get(agent_role, AGENT_SYSTEM_PROMPTS["general"])
    agent_info = AGENTS.get(agent_role)
    
    return {
        "agent_role": agent_role,
        "agent_name": agent_info.name if agent_info else "通用助手",
        "system_prompt": system_prompt
    }


app = FastAPI(title="LLM Assistant Pro", version="2.0.0", description="企业级AI智能助手平台")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/models", response_model=List[ModelInfo])
async def get_available_models():
    models = [
        ModelInfo(name="gpt-3.5-turbo", provider="OpenAI", available=openai_client is not None, max_tokens=4096),
        ModelInfo(name="gpt-4", provider="OpenAI", available=openai_client is not None, max_tokens=8192),
        ModelInfo(name="claude-3-sonnet", provider="Anthropic", available=anthropic_client is not None, max_tokens=200000),
        ModelInfo(name="claude-3-opus", provider="Anthropic", available=anthropic_client is not None, max_tokens=200000)
    ]
    return models


@app.get("/agents", response_model=List[AgentInfo])
async def get_available_agents():
    return list(AGENTS.values())


@app.get("/stats", response_model=SystemStats)
async def get_system_stats():
    return conversation_store.get_stats()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    agent_info = await multi_agent_orchestrator(request.message, request.agent_role)
    effective_prompt = request.system_prompt or agent_info["system_prompt"]
    
    if not request.conversation_id:
        conversation = conversation_store.create_conversation(
            model=request.model,
            system_prompt=effective_prompt,
            agent_role=agent_info["agent_role"]
        )
        conversation_id = conversation.conversation_id
    else:
        conversation = conversation_store.get_conversation(request.conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conversation_id = request.conversation_id
        if effective_prompt:
            conversation.system_prompt = effective_prompt

    conversation_store.add_message(conversation_id, "user", request.message)
    messages = [{"role": m.role, "content": m.content} for m in conversation.messages]
    
    tools = AVAILABLE_TOOLS if settings.enable_tool_calling else None
    
    try:
        if conversation.model.startswith("claude"):
            result = await call_anthropic(
                messages,
                model=conversation.model.replace("claude-3-", ""),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                system_prompt=conversation.system_prompt
            )
        else:
            result = await call_openai(
                messages,
                model=conversation.model if conversation.model != "openai" else settings.openai_model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                system_prompt=conversation.system_prompt,
                tools=tools
            )
        
        if result.get("tool_calls") and settings.enable_tool_calling:
            tool_results = []
            for tool_call in result["tool_calls"]:
                func_name = tool_call.function.name
                if func_name in TOOL_FUNCTIONS:
                    args = json.loads(tool_call.function.arguments)
                    tool_result = TOOL_FUNCTIONS[func_name](**args)
                    conversation_store.increment_tool_calls()
                    tool_results.append({
                        "role": "tool",
                        "content": tool_result,
                        "tool_call_id": tool_call.id,
                        "name": func_name
                    })
            
            if tool_results:
                for tool_result in tool_results:
                    conversation_store.add_message(
                        conversation_id, 
                        "tool", 
                        tool_result["content"],
                        tool_call_id=tool_result["tool_call_id"]
                    )
                
                messages.append({"role": "assistant", "content": "", "tool_calls": result["tool_calls"]})
                messages.extend(tool_results)
                
                if conversation.model.startswith("claude"):
                    final_result = await call_anthropic(
                        messages,
                        model=conversation.model.replace("claude-3-", ""),
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        system_prompt=conversation.system_prompt
                    )
                else:
                    final_result = await call_openai(
                        messages,
                        model=conversation.model if conversation.model != "openai" else settings.openai_model,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        system_prompt=conversation.system_prompt
                    )
                result = final_result
        
        conversation_store.add_message(conversation_id, "assistant", result["content"], agent_name=agent_info["agent_name"])
        
        return ChatResponse(
            conversation_id=conversation_id,
            message=result["content"],
            total_tokens=result.get("total_tokens"),
            model=result.get("model"),
            finish_reason=result.get("finish_reason"),
            agent_name=agent_info["agent_name"]
        )
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    agent_info = await multi_agent_orchestrator(request.message, request.agent_role)
    effective_prompt = request.system_prompt or agent_info["system_prompt"]
    
    if not request.conversation_id:
        conversation = conversation_store.create_conversation(
            model=request.model,
            system_prompt=effective_prompt,
            agent_role=agent_info["agent_role"]
        )
        conversation_id = conversation.conversation_id
    else:
        conversation = conversation_store.get_conversation(request.conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conversation_id = request.conversation_id
        if effective_prompt:
            conversation.system_prompt = effective_prompt

    conversation_store.add_message(conversation_id, "user", request.message)
    messages = [{"role": m.role, "content": m.content} for m in conversation.messages]

    async def generate():
        full_response = ""
        try:
            if conversation.model.startswith("claude"):
                async for chunk in stream_anthropic(
                    messages,
                    model=conversation.model.replace("claude-3-", ""),
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    system_prompt=conversation.system_prompt
                ):
                    full_response += chunk
                    yield chunk
            else:
                async for chunk in stream_openai(
                    messages,
                    model=conversation.model if conversation.model != "openai" else settings.openai_model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    system_prompt=conversation.system_prompt
                ):
                    full_response += chunk
                    yield chunk
            
            conversation_store.add_message(conversation_id, "assistant", full_response, agent_name=agent_info["agent_name"])
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            yield f"Error: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/conversations")
async def list_conversations(limit: Optional[int] = 10, offset: Optional[int] = 0):
    conversations = conversation_store.list_conversations()
    paginated = conversations[offset:offset + limit]
    return {"conversations": paginated, "total": len(conversations), "limit": limit, "offset": offset}


@app.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    conversation = conversation_store.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.put("/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, system_prompt: Optional[str] = None, model: Optional[str] = None, agent_role: Optional[str] = None):
    conversation = conversation_store.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if system_prompt:
        conversation.system_prompt = system_prompt
    if model:
        conversation.model = model
    if agent_role:
        conversation.agent_role = agent_role
    conversation.updated_at = datetime.now()
    
    return {"message": "Conversation updated successfully", "conversation": conversation}


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    success = conversation_store.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation deleted successfully"}


@app.delete("/conversations")
async def clear_conversations(confirm: bool = False):
    if not confirm:
        raise HTTPException(status_code=400, detail="Please confirm with ?confirm=true")
    conversation_store.clear_all()
    return {"message": "All conversations cleared successfully"}


@app.post("/summarize/{conversation_id}")
async def summarize_conversation(conversation_id: str):
    conversation = conversation_store.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages_text = "\n".join([f"{m.role}: {m.content}" for m in conversation.messages])
    summary_prompt = f"请总结以下对话：\n\n{messages_text}\n\n总结要求：\n1. 简洁明了，不超过200字\n2. 包含对话的主要内容和结论\n3. 使用中文回复"
    
    summary_messages = [{"role": "user", "content": summary_prompt}]
    
    try:
        if conversation.model.startswith("claude"):
            result = await call_anthropic(summary_messages, model=conversation.model.replace("claude-3-", ""), temperature=0.3, max_tokens=500)
        else:
            result = await call_openai(summary_messages, model=conversation.model if conversation.model != "openai" else settings.openai_model, temperature=0.3, max_tokens=500)
        
        return {"summary": result["content"], "conversation_id": conversation_id}
    except Exception as e:
        logger.error(f"Summarize error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/analyze")
async def analyze_text(text: str = Form(...), analysis_type: str = Form("sentiment")):
    prompts = {
        "sentiment": f"分析以下文本的情感倾向：\n\n{text}\n\n请返回：正面、负面或中性，并简要说明原因。",
        "keywords": f"提取以下文本的关键词：\n\n{text}\n\n请列出最重要的5-10个关键词。",
        "summary": f"总结以下文本：\n\n{text}\n\n请用简洁的语言总结主要内容。",
        "translation": f"将以下文本翻译成中文：\n\n{text}",
        "grammar": f"检查以下文本的语法错误：\n\n{text}\n\n请指出错误并给出修正建议。"
    }
    
    if analysis_type not in prompts:
        raise HTTPException(status_code=400, detail=f"Unknown analysis type: {analysis_type}. Available: {list(prompts.keys())}")
    
    messages = [{"role": "user", "content": prompts[analysis_type]}]
    
    try:
        result = await call_openai(messages, temperature=0.3, max_tokens=500)
        return {"analysis_type": analysis_type, "result": result["content"]}
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


HTML_CONTENT = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Assistant Pro - 企业级AI智能助手</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
            min-height: 100vh; display: flex; justify-content: center; padding: 20px; color: #333;
        }
        .container { width: 100%; max-width: 950px; background: #fff; border-radius: 24px; box-shadow: 0 30px 100px rgba(0,0,0,0.4); overflow: hidden; display: flex; flex-direction: column; height: 95vh; }
        .header { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); padding: 22px 28px; color: white; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-size: 24px; font-weight: 700; display: flex; align-items: center; gap: 12px; }
        .header h1 span { font-size: 28px; }
        .header-controls { display: flex; gap: 15px; align-items: center; }
        .selector-group { display: flex; gap: 8px; align-items: center; }
        .selector-group label { font-size: 13px; opacity: 0.9; }
        .selector-group select {
            padding: 9px 14px; border: none; border-radius: 10px; font-size: 13px;
            background: rgba(255,255,255,0.2); color: white; cursor: pointer;
        }
        .selector-group select option { color: #333; }
        .chat-container { flex: 1; overflow-y: auto; padding: 22px; display: flex; flex-direction: column; }
        .message { max-width: 75%; margin-bottom: 20px; padding: 15px 20px; border-radius: 22px; line-height: 1.65; animation: fadeIn 0.35s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
        .user-message { align-self: flex-end; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); color: white; border-bottom-right-radius: 6px; }
        .assistant-message { align-self: flex-start; background: #f1f5f9; color: #1e293b; border-bottom-left-radius: 6px; }
        .agent-badge { display: inline-block; padding: 3px 10px; background: rgba(99,102,241,0.15); color: #6366f1; border-radius: 12px; font-size: 11px; margin-bottom: 6px; }
        .tool-message { align-self: center; background: #dcfce7; color: #166534; font-size: 13px; padding: 12px 18px; border-radius: 14px; max-width: 60%; }
        .message-timestamp { font-size: 11px; opacity: 0.5; margin-top: 6px; text-align: right; }
        .input-container { padding: 18px 22px; border-top: 1px solid #e2e8f0; display: flex; gap: 14px; background: #fafbfc; }
        .input-container input {
            flex: 1; padding: 15px 22px; border: 2px solid #e2e8f0; border-radius: 28px;
            font-size: 15px; transition: all 0.3s; outline: none; background: white;
        }
        .input-container input:focus { border-color: #6366f1; box-shadow: 0 0 0 4px rgba(99,102,241,0.1); }
        .input-container button {
            padding: 15px 32px; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white; border: none; border-radius: 28px; font-size: 15px; font-weight: 600;
            cursor: pointer; transition: all 0.25s; display: flex; align-items: center; gap: 8px;
        }
        .input-container button:hover { transform: translateY(-2px); box-shadow: 0 6px 25px rgba(99,102,241,0.4); }
        .typing-indicator { display: flex; gap: 6px; padding: 15px 20px; }
        .typing-dot { width: 10px; height: 10px; background: #94a3b8; border-radius: 50%; animation: typing 1.4s infinite ease-in-out both; }
        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }
        .typing-dot:nth-child(3) { animation-delay: 0s; }
        @keyframes typing { 0%, 80%, 100% { transform: scale(0.6); opacity: 0.5; } 40% { transform: scale(1); opacity: 1; } }
        .empty-state { flex: 1; display: flex; flex-direction: column; justify-content: center; align-items: center; color: #64748b; text-align: center; }
        .empty-state .icon { font-size: 72px; margin-bottom: 22px; }
        .empty-state h3 { font-size: 22px; margin-bottom: 12px; color: #334155; }
        .empty-state p { font-size: 15px; line-height: 1.7; max-width: 320px; color: #64748b; }
        .empty-state .features { display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px; margin-top: 35px; width: 100%; max-width: 420px; }
        .empty-state .feature { padding: 18px; background: #f8fafc; border-radius: 16px; display: flex; flex-direction: column; align-items: center; gap: 10px; }
        .empty-state .feature span { font-size: 28px; }
        .empty-state .feature label { font-size: 14px; color: #475569; font-weight: 500; }
        .system-prompt-btn {
            position: absolute; bottom: 110px; right: 30px;
            padding: 11px 18px; background: #fff; border: 1px solid #e2e8f0;
            border-radius: 22px; font-size: 13px; cursor: pointer;
            display: flex; align-items: center; gap: 7px;
            box-shadow: 0 3px 15px rgba(0,0,0,0.08); transition: all 0.25s;
        }
        .system-prompt-btn:hover { box-shadow: 0 5px 20px rgba(0,0,0,0.12); border-color: #6366f1; }
        .modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.6); justify-content: center; align-items: center; z-index: 1000; }
        .modal.show { display: flex; }
        .modal-content { background: white; border-radius: 20px; padding: 30px; width: 90%; max-width: 520px; animation: modalIn 0.25s ease; }
        @keyframes modalIn { from { opacity: 0; transform: scale(0.95); } to { opacity: 1; transform: scale(1); } }
        .modal-content h3 { margin-bottom: 18px; color: #1e293b; font-size: 18px; }
        .modal-content textarea {
            width: 100%; height: 160px; padding: 14px; border: 1px solid #e2e8f0;
            border-radius: 12px; font-family: monospace; font-size: 14px; resize: vertical; outline: none;
        }
        .modal-content textarea:focus { border-color: #6366f1; }
        .modal-content .actions { display: flex; justify-content: flex-end; gap: 12px; margin-top: 22px; }
        .modal-content button { padding: 11px 24px; border: none; border-radius: 10px; cursor: pointer; font-size: 14px; font-weight: 500; }
        .modal-content button.cancel { background: #f1f5f9; color: #64748b; }
        .modal-content button.save { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); color: white; }
        @media (max-width: 500px) {
            .container { height: 100vh; border-radius: 0; }
            .message { max-width: 92%; }
            .header { flex-direction: column; gap: 12px; align-items: flex-start; }
            .header h1 { font-size: 20px; }
            .header-controls { width: 100%; justify-content: space-between; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><span>🚀</span> LLM Assistant Pro</h1>
            <div class="header-controls">
                <div class="selector-group">
                    <label>模型:</label>
                    <select id="modelSelector">
                        <option value="openai">GPT-3.5</option>
                        <option value="gpt-4">GPT-4</option>
                        <option value="claude-3-sonnet">Claude 3 Sonnet</option>
                        <option value="claude-3-opus">Claude 3 Opus</option>
                    </select>
                </div>
                <div class="selector-group">
                    <label>角色:</label>
                    <select id="agentSelector">
                        <option value="general">通用助手</option>
                        <option value="analyst">数据分析</option>
                        <option value="writer">文案创作</option>
                        <option value="code">代码助手</option>
                        <option value="translator">翻译专家</option>
                    </select>
                </div>
            </div>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="empty-state" id="emptyState">
                <div class="icon">🤖</div>
                <h3>欢迎使用企业级AI智能助手</h3>
                <p>支持多Agent协作、工具调用和长链推理，为您提供专业的AI服务体验</p>
                <div class="features">
                    <div class="feature"><span>🔄</span><label>多轮对话</label></div>
                    <div class="feature"><span>🛠️</span><label>工具调用</label></div>
                    <div class="feature"><span>👥</span><label>多Agent协作</label></div>
                    <div class="feature"><span>🧠</span><label>长链推理</label></div>
                </div>
            </div>
        </div>
        
        <button class="system-prompt-btn" id="systemPromptBtn">⚙️ 系统提示词</button>
        
        <div class="input-container">
            <input type="text" id="messageInput" placeholder="输入您的问题..." autocomplete="off"
                onkeydown="if(event.keyCode===13) sendMessage()">
            <button id="sendButton" onclick="sendMessage()">
                <span>发送</span>
                <span id="loadingIndicator" style="display:none;">⏳</span>
            </button>
        </div>
    </div>
    
    <div class="modal" id="systemPromptModal">
        <div class="modal-content">
            <h3>系统提示词设置</h3>
            <textarea id="systemPromptInput" placeholder="输入系统提示词，用于指导AI的行为..."></textarea>
            <div class="actions">
                <button class="cancel" onclick="closeModal()">取消</button>
                <button class="save" onclick="saveSystemPrompt()">保存</button>
            </div>
        </div>
    </div>

    <script>
        const chatContainer = document.getElementById('chatContainer');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const loadingIndicator = document.getElementById('loadingIndicator');
        const emptyState = document.getElementById('emptyState');
        const modelSelector = document.getElementById('modelSelector');
        const agentSelector = document.getElementById('agentSelector');
        const systemPromptBtn = document.getElementById('systemPromptBtn');
        const systemPromptModal = document.getElementById('systemPromptModal');
        const systemPromptInput = document.getElementById('systemPromptInput');
        
        let currentConversationId = null;
        let isTyping = false;
        let currentSystemPrompt = '';

        function showModal() { systemPromptModal.classList.add('show'); }
        function closeModal() { systemPromptModal.classList.remove('show'); }
        function saveSystemPrompt() { currentSystemPrompt = systemPromptInput.value; closeModal(); }
        systemPromptBtn.addEventListener('click', showModal);

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message || isTyping) return;

            isTyping = true;
            messageInput.value = '';
            sendButton.querySelector('span:first-child').style.display = 'none';
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
                        conversation_id: currentConversationId || undefined,
                        model: modelSelector.value,
                        agent_role: agentSelector.value,
                        system_prompt: currentSystemPrompt || undefined,
                        temperature: 0.7,
                        max_tokens: 4096
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
                if (data.conversations.length > 0) currentConversationId = data.conversations[0].conversation_id;
            } catch (error) {
                typingIndicator?.remove();
                addMessage('抱歉，服务暂时不可用，请检查API密钥配置。', 'assistant');
            } finally {
                isTyping = false;
                sendButton.querySelector('span:first-child').style.display = 'inline';
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
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, reload=True)