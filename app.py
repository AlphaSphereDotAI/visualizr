import os

from chatacter.model import generate_video
from chatacter.settings import get_settings
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(debug=True)
settings = get_settings()


@app.get("/")
async def is_alive():
    return JSONResponse(
        content={
            "message": "Chatacter Video Generator is alive!",
        },
    )


@app.get("/get_settings")
async def get_settings():
    return settings.model_dump()


@app.post("/set_audio")
async def set_audio(file: UploadFile):
    with open(f"{settings.assets.audio}AUDIO.wav", "wb") as audio:
        audio.write(await file.read())


@app.get("/get_video")
def get_video(character: str):
    time = generate_video(character)
    files = os.listdir("./assets/results")
    files.sort(reverse=True)
    print(f"File in {settings.assets.video}{files[0]}")
    return FileResponse(
        path=settings.assets.video + files[0],
        media_type="video/mp4",
        filename="VIDEO.mp4",
        headers={"time": time},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=True)
