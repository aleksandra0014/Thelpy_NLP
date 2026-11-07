"""python -m thelpy_api  ->  uvicorn na THELPY_API_HOST:THELPY_API_PORT (domyślnie 127.0.0.1:8000)."""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("thelpy_api.app:app", host=os.getenv("THELPY_API_HOST", "127.0.0.1"),
                port=int(os.getenv("THELPY_API_PORT", "8000")),
                reload=os.getenv("THELPY_API_RELOAD", "") == "1")
