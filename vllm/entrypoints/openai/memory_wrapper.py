"""
Memory-Aware API Wrapper for vLLM

This wrapper:
1. Receives requests with memory_text content parts containing raw text to encode
2. Encodes them using HuggingFace model (QFormer + Projector)
3. Converts to base64 memory_embeds and forwards to vLLM API
4. Returns the response to client

Supported input format:
    {
        "type": "memory_text",
        "memory_text": {
            "text": "<html>...</html>",
            "is_memory": true  # optional, default true
        }
    }

Output format (forwarded to vLLM):
    {
        "type": "memory_embeds",
        "memory_embeds": "<base64-encoded-tensor>"
    }

Usage:
    python -m vllm.entrypoints.openai.memory_wrapper \
        --model-path /path/to/qwen3-memory-model \
        --vllm-api-base http://localhost:8000 \
        --port 8001
"""

import argparse
import asyncio
import base64
import hashlib
import io
import logging
from collections import OrderedDict
from typing import Any, AsyncIterator, Optional

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU cache for text -> base64 mapping."""

    def __init__(self, max_size: int = 4000):
        self.max_size = max_size
        self.cache: OrderedDict[str, str] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _hash_key(self, text: str) -> str:
        """Create a hash key from text to save memory."""
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[str]:
        """Get cached base64 for text, or None if not cached."""
        key = self._hash_key(text)
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, text: str, b64: str) -> None:
        """Cache text -> base64 mapping."""
        key = self._hash_key(text)
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove oldest
                self.cache.popitem(last=False)
            self.cache[key] = b64

    def stats(self) -> dict:
        """Return cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2%}",
        }


class MemoryEncoder:
    """HuggingFace-based memory encoder using QFormer + Projector."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = None
        self.tokenizer = None
        self.model_path = model_path

    def load(self):
        """Load the HuggingFace model for encoding."""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info(f"Loading memory encoder from {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

        logger.info("Memory encoder loaded successfully")

    @torch.inference_mode()
    def encode(
        self,
        memory_input_ids: torch.Tensor,
        memory_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode memory tokens into embeddings.

        Args:
            memory_input_ids: (num_memories, seq_len)
            memory_attention_mask: (num_memories, seq_len)

        Returns:
            memory_embeds: (num_memories, hidden_size)
        """
        memory_input_ids = memory_input_ids.to(self.device)
        memory_attention_mask = memory_attention_mask.to(self.device)

        # Use the model's encode method
        memory_embeds = self.model.encode(
            input_ids=memory_input_ids,
            attention_mask=memory_attention_mask,
        )

        return memory_embeds.cpu()

    def encode_from_texts(self, texts: list[str]) -> torch.Tensor:
        """
        Encode a list of text strings into memory embeddings.

        Args:
            texts: List of text strings to encode as memories

        Returns:
            memory_embeds: (num_memories, hidden_size)
        """
        if not texts:
            return None

        # Tokenize all texts
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return self.encode(
            memory_input_ids=encodings["input_ids"],
            memory_attention_mask=encodings["attention_mask"],
        )


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert a PyTorch tensor to base64 string."""
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_tensor(b64_str: str) -> torch.Tensor:
    """Convert a base64 string back to PyTorch tensor."""
    buffer = io.BytesIO(base64.b64decode(b64_str))
    return torch.load(buffer, weights_only=True)


class MemoryWrapperConfig(BaseModel):
    """Configuration for the memory wrapper."""
    model_path: str
    vllm_api_base: str = "http://localhost:8000"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"


class MemoryWrapperServer:
    """
    FastAPI server that wraps vLLM API with memory encoding capability.
    """

    def __init__(self, config: MemoryWrapperConfig, cache_size: int = 4000):
        self.config = config
        self.app = FastAPI(title="Memory-Aware vLLM Wrapper")
        self.encoder: Optional[MemoryEncoder] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.cache = LRUCache(max_size=cache_size)

        self._setup_routes()

    def _setup_routes(self):
        @self.app.on_event("startup")
        async def startup():
            # Load encoder
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            self.encoder = MemoryEncoder(
                model_path=self.config.model_path,
                device=self.config.device,
                torch_dtype=dtype_map.get(self.config.torch_dtype, torch.bfloat16),
            )
            self.encoder.load()

            # Create HTTP client for vLLM API
            self.http_client = httpx.AsyncClient(
                base_url=self.config.vllm_api_base,
                timeout=httpx.Timeout(600.0),  # 10 min timeout for long generations
            )

        @self.app.on_event("shutdown")
        async def shutdown():
            if self.http_client:
                await self.http_client.aclose()

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            return await self._handle_chat_completions(request)

        @self.app.post("/v1/completions")
        async def completions(request: Request):
            return await self._handle_completions(request)

        @self.app.get("/health")
        async def health():
            return {"status": "ok", "encoder_loaded": self.encoder is not None}

        @self.app.get("/v1/models")
        async def list_models():
            # Proxy to vLLM
            response = await self.http_client.get("/v1/models")
            return JSONResponse(content=response.json())

        @self.app.get("/cache/stats")
        async def cache_stats():
            return self.cache.stats()

    async def _handle_chat_completions(self, request: Request) -> JSONResponse:
        """Handle /v1/chat/completions with memory encoding."""
        body = await request.json()

        # Transform messages: convert memory_text to memory_embeds
        messages = body.get("messages", [])
        transformed_messages = []

        for msg in messages:
            transformed_msg = self._transform_message(msg)
            transformed_messages.append(transformed_msg)

        body["messages"] = transformed_messages

        # Forward to vLLM
        stream = body.get("stream", False)

        if stream:
            return StreamingResponse(
                self._stream_response(body),
                media_type="text/event-stream",
            )
        else:
            response = await self.http_client.post(
                "/v1/chat/completions",
                json=body,
            )
            return JSONResponse(content=response.json())

    async def _handle_completions(self, request: Request) -> JSONResponse:
        """Handle /v1/completions with memory encoding."""
        body = await request.json()

        # For completions API, just forward (memory_text not typically used here)
        stream = body.get("stream", False)

        if stream:
            return StreamingResponse(
                self._stream_response(body, endpoint="/v1/completions"),
                media_type="text/event-stream",
            )
        else:
            response = await self.http_client.post(
                "/v1/completions",
                json=body,
            )
            return JSONResponse(content=response.json())

    def _transform_message(self, msg: dict[str, Any]) -> dict[str, Any]:
        """
        Transform a message by converting memory_text parts to memory_embeds.

        Input format:
            {
                "type": "memory_text",
                "memory_text": {
                    "text": "<html>...</html>",
                    "is_memory": true  # optional
                }
            }

        Output format:
            {
                "type": "memory_embeds",
                "memory_embeds": "<base64-encoded-tensor>"
            }
        """
        content = msg.get("content")

        # String content - no transformation needed
        if isinstance(content, str):
            return msg

        # List content - transform each part
        if isinstance(content, list):
            new_content = []
            for part in content:
                transformed_part = self._transform_content_part(part)
                new_content.append(transformed_part)

            return {**msg, "content": new_content}

        return msg

    def _transform_content_part(self, part: dict[str, Any]) -> dict[str, Any]:
        """
        Transform a single content part.

        If it's a memory_text part, encode and convert to memory_embeds.
        Uses LRU cache to avoid re-encoding same text.
        Otherwise, return as-is.
        """
        if not isinstance(part, dict):
            return part

        part_type = part.get("type")

        if part_type != "memory_text":
            return part

        # Extract memory_text content
        memory_text_data = part.get("memory_text", {})

        # Handle both dict and string formats
        if isinstance(memory_text_data, str):
            text = memory_text_data
            is_memory = True
        else:
            text = memory_text_data.get("text", "")
            is_memory = memory_text_data.get("is_memory", True)

        # If is_memory is False, convert to regular text
        if not is_memory:
            return {"type": "text", "text": text}

        # Check cache first
        cached_b64 = self.cache.get(text)
        if cached_b64 is not None:
            return {
                "type": "memory_embeds",
                "memory_embeds": cached_b64,
            }

        # Cache miss - encode the text
        memory_embed = self.encoder.encode_from_texts([text])

        if memory_embed is None:
            # Fallback to text if encoding fails
            return {"type": "text", "text": text}

        # Convert to base64 and cache
        memory_b64 = tensor_to_base64(memory_embed)
        self.cache.put(text, memory_b64)

        return {
            "type": "memory_embeds",
            "memory_embeds": memory_b64,
        }

    async def _stream_response(
        self,
        body: dict[str, Any],
        endpoint: str = "/v1/chat/completions",
    ) -> AsyncIterator[bytes]:
        """Stream response from vLLM."""
        async with self.http_client.stream(
            "POST",
            endpoint,
            json=body,
        ) as response:
            async for chunk in response.aiter_bytes():
                yield chunk


def create_app(
    model_path: str,
    vllm_api_base: str = "http://localhost:8000",
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    cache_size: int = 4000,
) -> FastAPI:
    """Create the FastAPI application."""
    config = MemoryWrapperConfig(
        model_path=model_path,
        vllm_api_base=vllm_api_base,
        device=device,
        torch_dtype=torch_dtype,
    )
    server = MemoryWrapperServer(config, cache_size=cache_size)
    return server.app


def main():
    parser = argparse.ArgumentParser(description="Memory-Aware vLLM API Wrapper")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the HuggingFace model for memory encoding",
    )
    parser.add_argument(
        "--vllm-api-base",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the vLLM API server",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the wrapper server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind the wrapper server",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for the encoder model",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for the encoder model",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=4000,
        help="LRU cache size for text -> base64 mapping (default: 4000)",
    )

    args = parser.parse_args()

    app = create_app(
        model_path=args.model_path,
        vllm_api_base=args.vllm_api_base,
        device=args.device,
        torch_dtype=args.torch_dtype,
        cache_size=args.cache_size,
    )

    logger.info(f"Starting Memory Wrapper Server on {args.host}:{args.port}")
    logger.info(f"Proxying to vLLM at {args.vllm_api_base}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
