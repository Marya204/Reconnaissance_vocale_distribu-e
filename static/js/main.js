document.addEventListener('DOMContentLoaded', () => {
    const recordBtn = document.getElementById('recordBtn');
    const processBtn = document.getElementById('processBtn');
    const audioUpload = document.getElementById('audioUpload');
    const audioPlayer = document.getElementById('audioPlayback');
    const predictionText = document.getElementById('predictionText');  // Nouvelle section pour afficher la prédiction
    
    let audioContext;
    let audioBuffer;
    
    // Enregistrement audio
    recordBtn.addEventListener('click', async () => {
        recordBtn.disabled = true;
        
        try {
            const response = await fetch('/record', { method: 'POST' });
            const data = await response.json();
            updateDisplay(data);  // Mise à jour de l'affichage avec la prédiction
        } catch (error) {
            console.error("Recording failed:", error);
        } finally {
            recordBtn.disabled = false;
        }
    });
    
    // Traitement du fichier téléchargé
    processBtn.addEventListener('click', async () => {
        if (!audioUpload.files.length) return;
        
        const formData = new FormData();
        formData.append('audio', audioUpload.files[0]);
        
        processBtn.disabled = true;
        
        try {
            const response = await fetch('/process', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            updateDisplay(data);  // Mise à jour de l'affichage avec la prédiction
        } catch (error) {
            console.error("Processing failed:", error);
        } finally {
            processBtn.disabled = false;
        }
    });
    
    // Met à jour l'affichage avec la prédiction
    function updateDisplay(data) {
        // Mise à jour du lecteur audio
        updateAudioPlayer(data.audio, data.sample_rate);
        
        // Affichage de la prédiction
        if (data.prediction) {
            predictionText.textContent = `Prediction: ${data.prediction}`;  // Affiche la prédiction retournée par le backend
        }
    }
    
    // Met à jour le lecteur audio
    function updateAudioPlayer(audioData, sampleRate) {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        
        const buffer = audioContext.createBuffer(1, audioData.length, sampleRate);
        buffer.getChannelData(0).set(new Float32Array(audioData));
        
        if (audioBuffer) {
            audioBuffer.stop();
        }
        
        const source = audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(audioContext.destination);
        source.start();
        audioBuffer = source;
        
        // Mise à jour du lecteur audio HTML5
        const wavBlob = audioToWavBlob(audioData, sampleRate);
        audioPlayer.src = URL.createObjectURL(wavBlob);
    }
    
    // Convertir le tableau audio en un blob WAV
    function audioToWavBlob(audioData, sampleRate) {
        const buffer = new ArrayBuffer(44 + audioData.length * 2);
        const view = new DataView(buffer);
        
        // Écriture de l'en-tête WAV
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        writeString(0, 'RIFF');
        view.setUint32(4, 32 + audioData.length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt '); 
        view.setUint32(16, 16, true); 
        view.setUint16(20, 1, true); 
        view.setUint16(22, 1, true); 
        view.setUint32(24, sampleRate, true); 
        view.setUint32(28, sampleRate * 2, true); 
        view.setUint16(32, 2, true); 
        view.setUint16(34, 16, true); 
        writeString(36, 'data'); 
        view.setUint32(40, audioData.length * 2, true);
        
        // Écrire les échantillons audio
        const floatTo16Bit = (sample) => {
            sample = Math.max(-1, Math.min(1, sample));
            return sample < 0 ? sample * 32768 : sample * 32767;
        };
        
        for (let i = 0; i < audioData.length; i++) {
            view.setInt16(44 + i * 2, floatTo16Bit(audioData[i]), true);
        }
        
        return new Blob([view], { type: 'audio/wav' });
    }
});
