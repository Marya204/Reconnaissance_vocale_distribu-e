from flask import render_template, jsonify

class AudioView:
    def home(self):
        return render_template("index.html")

    def transcription(self, text, confidence=None):
        return render_template("index.html", text=text, confidence=confidence)

    def error(self, message):
        return render_template("index.html", error=message)
