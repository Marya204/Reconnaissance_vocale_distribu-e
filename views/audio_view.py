from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

templates = Jinja2Templates(directory="templates")

class AudioView:
    def home(self, request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    def transcription(self, request: Request, text, confidence=None):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "text": text,
            "confidence": confidence
        })

    def error(self, request: Request, message):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": message
        })
    def audio_data(self, data):  # <-- Ajoute cette méthode
        return JSONResponse(content=data)