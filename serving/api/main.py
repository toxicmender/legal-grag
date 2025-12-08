"""
FastAPI application with routes for user queries, ingestion, retrieval, etc.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional
import uvicorn

from .routes import router
from ..config import ServerConfig
from ..session_manager import SessionManager


def create_app(config: Optional[ServerConfig] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        config: Optional ServerConfig instance.
        
    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="Legal Knowledge Graph API",
        description="API for legal knowledge graph system",
        version="1.0.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router)
    
    # Initialize session manager
    app.state.session_manager = SessionManager()
    app.state.config = config or ServerConfig()
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        # TODO: Initialize services (database, LLM clients, etc.)
        pass
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        # TODO: Cleanup resources
        pass
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Legal Knowledge Graph API",
            "version": "1.0.0",
            "status": "running"
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the FastAPI server.
    
    Args:
        host: Host to bind to.
        port: Port to bind to.
        reload: Enable auto-reload for development.
    """
    app = create_app()
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server(reload=True)

