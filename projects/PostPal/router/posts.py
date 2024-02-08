from typing import List
from fastapi import HTTPException, Depends, status, APIRouter
from sqlalchemy.orm import Session
from db import model, schema
from db.database import get_db

router = APIRouter(prefix="/posts", tags=["Posts"])
