from flask import render_template, jsonify

class AudioView:
    @staticmethod
    def home():
        return render_template('index.html')
    
    @staticmethod
    def audio_data(data):
        return jsonify(data)
    
    @staticmethod
    def error(message):
        return jsonify({'error': message})