# server.py - Auto-detect videos from folder

from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import requests

import os
import logging
import json
import cv2
import asyncio
import time
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
import numpy as np
import base64

# Import the real parking detection system
from parking_detection import ParkingDetectionSystem

ROOT_DIR = Path(__file__).parent

# Load environment file
if (ROOT_DIR / '.env.local').exists():
    load_dotenv(ROOT_DIR / '.env.local')
    print("Loaded .env.local for local development")
else:
    load_dotenv(ROOT_DIR / '.env')
    print("Loaded .env for production")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'parking_db')]

# Create the main app
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Initialize parking detection system
def initialize_parking_system():
    """Initialize parking detection system with model paths"""
    model_path = os.environ.get(
        'PARKING_MODEL_PATH',
        str(ROOT_DIR / 'models' / 'parking_model_train22.pkl')
    )
    csv_path = os.environ.get(
        'PARKING_CSV_PATH',
        str(ROOT_DIR / 'models' / 'yolo_features_train22.csv')
    )

    logger.info("Initializing parking system with:")
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  CSV path: {csv_path}")

    if not Path(model_path).exists():
        raise RuntimeError(f"Model file not found at {model_path}")
    
    system = ParkingDetectionSystem(model_path=model_path, csv_path=csv_path)
    logger.info("✓ Parking detection system initialized successfully")
    return system

parking_system = initialize_parking_system()

# Create videos directory
videos_dir = ROOT_DIR / "videos"
videos_dir.mkdir(exist_ok=True)

# AUTO-DETECT VIDEOS FROM FOLDER
def auto_detect_videos():
    """
    Automatically detect all video files in the videos folder.
    Supports: .mp4, .avi, .mov, .mkv
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
    available_videos = {}
    
    logger.info(f"Scanning videos directory: {videos_dir}")
    
    if not videos_dir.exists():
        logger.warning(f"Videos directory does not exist: {videos_dir}")
        return available_videos
    
    # Scan for video files
    for video_file in videos_dir.iterdir():
        if video_file.is_file() and video_file.suffix in video_extensions:
            # Use filename without extension as the display name
            display_name = video_file.stem.replace('_', ' ').replace('-', ' ')
            relative_path = f"videos/{video_file.name}"
            
            available_videos[display_name] = relative_path
            logger.info(f"  Found video: {display_name} -> {relative_path}")
    
    logger.info(f"Total videos found: {len(available_videos)}")
    return available_videos

# Get available videos (auto-detected)
AVAILABLE_VIDEOS = auto_detect_videos()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                if connection in self.active_connections:
                    self.active_connections.remove(connection)

manager = ConnectionManager()

# Pydantic Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class ViolationLog(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vehicle_id: int
    location: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration: float
    violation_type: str = "no_parking_zone"

class VideoFrameProcessRequest(BaseModel):
    video_name: str
    timestamp: float = 0.0
    alert_threshold: Optional[float] = 5.0

class FrameProcessRequest(BaseModel):
    frame_data: str
    video_name: Optional[str] = "unknown"
    alert_threshold: Optional[float] = 5.0

# Basic routes
@api_router.get("/")
async def root():
    return {"message": "Parking Detection System API"}

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    stats = parking_system.get_statistics()
    return {
        "status": "healthy",
        "parking_system": "active",
        "statistics": stats,
        "timestamp": datetime.utcnow().isoformat()
    }

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# Video routes
@api_router.get("/videos")
async def get_available_videos():
    """Get list of available videos (auto-detected)"""
    available: Dict[str, Dict[str, Any]] = {}
    for name, path in AVAILABLE_VIDEOS.items():
        full_path = ROOT_DIR / path
        available[name] = {
            "path": path,
            "exists": full_path.exists(),
            "size": full_path.stat().st_size if full_path.exists() else 0
        }
    return {
        "videos": available,
        "total": len(available)
    }

@api_router.get("/video/{video_name}")
@api_router.head("/video/{video_name}")
async def get_video_file(video_name: str, request: Request):
    """Serve video files with streaming support"""
    if video_name not in AVAILABLE_VIDEOS:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = ROOT_DIR / AVAILABLE_VIDEOS[video_name]
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file missing")

    headers = {
        "Content-Type": "video/mp4",
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-cache"
    }

    return FileResponse(
        video_path,
        headers=headers,
        filename=f"{video_name}.mp4"
    )

# Process video frame at specific timestamp
@api_router.post("/process-video-frame")
async def process_video_frame(request: VideoFrameProcessRequest):
    """
    Process a frame from video file at specific timestamp.
    This is Solution 3 - backend processes video directly.
    """
    try:
        start_time = time.time()
        
        video_name = request.video_name
        timestamp = request.timestamp
        alert_threshold = request.alert_threshold

        # Validate video exists
        if video_name not in AVAILABLE_VIDEOS:
            raise HTTPException(status_code=404, detail=f"Video '{video_name}' not found")

        video_path = ROOT_DIR / AVAILABLE_VIDEOS[video_name]
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video file missing: {video_path}")

        logger.info(f"Processing frame from {video_name} at {timestamp}s")

        # Open video and seek to timestamp
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame number from timestamp
        frame_number = int(timestamp * fps)
        
        # Ensure frame number is valid
        if frame_number >= total_frames:
            frame_number = total_frames - 1
        
        # Seek to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read frame
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise ValueError(f"Failed to read frame at timestamp {timestamp}s")

        logger.info(f"Successfully extracted frame: shape={frame.shape}")

        # Process frame with parking detection system
        result = parking_system.process_frame(
            frame,
            alert_threshold_seconds=alert_threshold
        )

        # Add processing metadata
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        result['video_name'] = video_name
        result['timestamp'] = timestamp
        result['frame_number'] = frame_number
        result['fps'] = fps

        logger.info(f"Frame processed in {processing_time:.2f}s - "
                   f"Prediction: {result.get('prediction')}, "
                   f"Vehicles: {len(result.get('vehicles', []))}, "
                   f"Violations: {result.get('total_violations', 0)}")

        # Broadcast new alerts via WebSocket
        if result.get('new_alerts'):
            await manager.broadcast({
                'type': 'new_alerts',
                'data': result['new_alerts'],
                'video_name': video_name,
                'timestamp': time.time()
            })
            # --- Trigger Pico 2 W buzzer ---
            pico_ip = "172.20.10.2"  # <-- replace with your Pico's IP
            try:
                requests.get(f"http://{pico_ip}/alarm", timeout=2)
                logger.info("Buzzer triggered for new alert (video frame).")
            except Exception as e:
                logger.warning(f"Failed to trigger Pico buzzer: {e}")
            # --------------------------------

        return result

    except Exception as e:
        logger.error(f"Error processing video frame: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Legacy: Process frame from base64 data
@api_router.post("/process-frame")
async def process_frame_endpoint(request: FrameProcessRequest):
    """Process a single frame from base64 data (legacy support)"""
    try:
        start_time = time.time()

        # Decode base64 frame data
        frame_data = request.frame_data
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]

        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Failed to decode frame")

        # Process frame
        result = parking_system.process_frame(
            frame,
            alert_threshold_seconds=request.alert_threshold
        )

        # Add metadata
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        result['video_name'] = request.video_name

        # Broadcast new alerts
        if result.get('new_alerts'):
            await manager.broadcast({
                'type': 'new_alerts',
                'data': result['new_alerts'],
                'timestamp': time.time()
            })
            # --- Trigger Pico 2 W buzzer ---
            pico_ip = "172.20.10.2"  # <-- replace with your Pico's IP
            try:
                requests.get(f"http://{pico_ip}/alarm", timeout=2)
                logger.info("Buzzer triggered for new alert (frame upload).")
            except Exception as e:
                logger.warning(f"Failed to trigger Pico buzzer: {e}")
            # --------------------------------

        return result

    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/reset-alerts")
async def reset_alerts():
    """Reset all parking violation alerts"""
    try:
        # Clear violations from database
        await db.violations.delete_many({})

        # Reset system state
        parking_system.reset_alerts()

        # Broadcast reset to clients
        await manager.broadcast({
            'type': 'alerts_reset',
            'timestamp': time.time()
        })

        return {"message": "Alerts reset successfully", "timestamp": time.time()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/statistics")
async def get_statistics():
    """Get current parking system statistics"""
    try:
        stats = parking_system.get_statistics()
        return {
            "statistics": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Violation routes
@api_router.post("/violations", response_model=ViolationLog)
async def log_violation(violation: ViolationLog):
    """Log a parking violation"""
    try:
        violation_dict = violation.dict()
        _ = await db.violations.insert_one(violation_dict)
        
        # Broadcast to connected clients
        await manager.broadcast({
            'type': 'new_violation',
            'data': violation_dict
        })
        
        return violation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/violations", response_model=List[ViolationLog])
async def get_violations(limit: int = 500):
    """Get all parking violations"""
    try:
        violations = await db.violations.find().sort("timestamp", -1).to_list(limit)
        return [ViolationLog(**violation) for violation in violations]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/violations/{violation_id}")
async def delete_violation(violation_id: str):
    """Delete a specific violation"""
    try:
        result = await db.violations.delete_one({"id": violation_id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Violation not found")
        return {"message": "Violation deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get('type') == 'ping':
                await websocket.send_text(json.dumps({'type': 'pong'}))
            elif message.get('type') == 'request_stats':
                stats = parking_system.get_statistics()
                await websocket.send_text(json.dumps({
                    'type': 'statistics',
                    'data': stats
                }))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Include router and middleware
app.include_router(api_router)

app.mount("/static", StaticFiles(directory=str(videos_dir)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("Parking Violation Alert System - Starting Up")
    logger.info("=" * 60)
    logger.info(f"Environment: {os.environ.get('ENVIRONMENT', 'production')}")
    logger.info(f"Videos directory: {videos_dir}")
    logger.info(f"Available videos: {len(AVAILABLE_VIDEOS)}")
    for name in AVAILABLE_VIDEOS.keys():
        logger.info(f"  - {name}")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_db_client():
    logger.info("Shutting down database connection...")
    client.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get('PORT', 8001))
    print(f"Starting server on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)