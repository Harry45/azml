from fastapi import FastAPI
from db import model
from db.database import engine, SessionLocal

app = FastAPI()
model.Base.metadata.create_all(bind=engine)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
