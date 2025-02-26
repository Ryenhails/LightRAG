from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.llm.siliconcloud import siliconcloud_embedding
from lightrag.utils import EmbeddingFunc
import numpy as np
from typing import Optional
import asyncio
import nest_asyncio

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

DEFAULT_RAG_DIR = "small-scale_experiment"
app = FastAPI(title="LightRAG API", description="API for RAG operations")

# Configure working directory
WORKING_DIR = os.environ.get("RAG_DIR", f"{DEFAULT_RAG_DIR}")
print(f"WORKING_DIR: {WORKING_DIR}")

LLM_MODEL = os.environ.get("LLM_MODEL", "/mimer/NOBACKUP/groups/naiss2025-22-84/llm_models/deepseekr1-distill-llama-70B")
print(f"LLM_MODEL: {LLM_MODEL}")

BASE_URL = os.environ.get("BASE_URL", "http://localhost:1234/v1")
print(f"BASE_URL: {BASE_URL}")

API_KEY = os.environ.get("API_KEY", "EMPTY")
print(f"API_KEY: {API_KEY}")

SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "sk-uyhsrhunyracecldhzkrtqitmfjpndmnrcxjhvsduiegzhfs")
print(f"SILICONFLOW_API_KEY: {SILICONFLOW_API_KEY}")

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# LLM model function

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        model=LLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=BASE_URL,
        api_key=API_KEY,
        **kwargs,
    )


# Embedding function


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await siliconcloud_embedding(
        texts,
        model="BAAI/bge-large-en-v1.5",
        api_key=SILICONFLOW_API_KEY,
        max_token_size=512,
    )


# function test
async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("embedding_func: ", result)


asyncio.run(test_funcs())

# Initialize RAG instance
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024, max_token_size=512, func=embedding_func
    ),
)


# Data models


class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    only_need_context: bool = False


class InsertRequest(BaseModel):
    text: str


class Response(BaseModel):
    status: str
    data: Optional[str] = None
    message: Optional[str] = None


# API routes


@app.post("/query", response_model=Response)
async def query_endpoint(request: QueryRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: rag.query(
                request.query,
                param=QueryParam(
                    mode=request.mode, only_need_context=request.only_need_context
                ),
            ),
        )
        return Response(status="success", data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/insert", response_model=Response)
async def insert_endpoint(request: InsertRequest):
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: rag.insert(request.text))
        return Response(status="success", message="Text inserted successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/insert_file", response_model=Response)
async def insert_file(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        # Read file content
        try:
            content = file_content.decode("utf-8")
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try other encodings
            content = file_content.decode("gbk")
        # Insert file content
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: rag.insert(content))

        return Response(
            status="success",
            message=f"File content from {file.filename} inserted successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8020)

# Usage example
# To run the server, use the following command in your terminal:
# python 'examples/lightrag_api_vllm_compatible_demo.py'

# Example requests:
# 1. Query:
# curl -X POST "http://127.0.0.1:8020/query" -H "Content-Type: application/json" -d '{"query": "your query here", "mode": "hybrid"}'
# curl.exe -X POST "http://127.0.0.1:8020/query" -H "Content-Type: application/json" -d '{"query": "What is a principal type in the context of type inference?", "mode": "hybrid"}'

# 2. Insert text:
# curl -X POST "http://127.0.0.1:8020/insert" -H "Content-Type: application/json" -d '{"text": "your text here"}'

# 3. Insert file:
# curl -X POST "http://127.0.0.1:8020/insert_file" -H "Content-Type: multipart/form-data" -F "file=@path/to/your/file.txt"

#curl.exe -X POST "http://127.0.0.1:8020/insert_file" -H "Content-Type: multipart/form-data" -F "file=@E:\Master Thesis\lightrag\LightRAG\evaluation_data\cs_min_context.txt"

# 4. Health check:
# curl -X GET "http://127.0.0.1:8020/health"
