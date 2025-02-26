import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.siliconcloud import siliconcloud_embedding
from lightrag.utils import EmbeddingFunc
import numpy as np

WORKING_DIR = "./clue_2_experiment_Qwen2.5-72B"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "Qwen2.5-72B",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="None",
        base_url="http://localhost:1234/v1",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await siliconcloud_embedding(
        texts,
        model="BAAI/bge-m3",
        api_key="sk-uyhsrhunyracecldhzkrtqitmfjpndmnrcxjhvsduiegzhfs",
        max_token_size=8096,
    )


# function test
async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("embedding_func: ", result)


asyncio.run(test_funcs())


rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024, max_token_size=512, func=embedding_func
    ),
    enable_llm_cache = True
)


# with open("small-scale_experiment/cs_min_context.txt", encoding='utf-8') as f:
#      rag.insert(f.read())

# Perform local search

print(
    rag.query("I am a beginner, please give me a path to learn programming", param=QueryParam(mode="local"))
 )
#
print(
    rag.query("I am a beginner, please give me a path to learn programming", param=QueryParam(mode="local_with_clues"))
)
#
#
# # Perform global search
#
print(
     rag.query("I am a beginner, please give me a path to learn programming", param=QueryParam(mode="global"))
 )
#
print(
    rag.query("I am a beginner, please give me a path to learn programming", param=QueryParam(mode="global_with_clues"))
)
#
# # Perform hybrid search
print(
     rag.query("I am a beginner, please give me a path to learn programming", param=QueryParam(mode="hybrid"))
 )

print(
    rag.query("I am a beginner, please give me a path to learn programming", param=QueryParam(mode="hybrid_with_clues"))
)
