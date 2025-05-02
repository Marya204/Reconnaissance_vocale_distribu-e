from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from controllers.audio_controller import AudioController # type: ignore
from views.audio_view import AudioView # type: ignore

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'tmp_uploads'
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

controller = AudioController()
view = AudioView()

@app.route('/')
def home():
    return view.home()

@app.route('/record', methods=['POST'])
def record_audio():
    try:
        data = controller.record_and_process()
        return view.audio_data(data)
    except Exception as e:
        return view.error(str(e))

@app.route('/process', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return view.error("No audio file uploaded")
    
    file = request.files['audio']
    if file.filename == '':
        return view.error("No file selected")
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            data = controller.process_file(filepath)
            return view.audio_data(data)
        except Exception as e:
            return view.error(str(e))
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True)