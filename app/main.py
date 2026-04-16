"""
Copyright (c) 2026 Baechul Kim
All rights reserved.

Author: Baechul Kim <baechul@gmail.com>
Date: April 15, 2026
Description: Main entry point. 
License: MIT
"""

from fastapi import FastAPI
from app.api.endpoints import router

app = FastAPI()
app.include_router(router)
