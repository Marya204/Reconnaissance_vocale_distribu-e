document.addEventListener('DOMContentLoaded', () => {
    const recordBtn = document.getElementById('recordBtn');
    const processBtn = document.getElementById('processBtn');
    const audioUpload = document.getElementById('audioUpload');
    const audioPlayer = document.getElementById('audioPlayback');
    const spectrogramImg = document.getElementById('spectrogram');
    const mfccCanvas = document.getElementById('mfccChart');
    
    let audioContext;
    let audioBuffer;
    
    // Record audio
    recordBtn.addEventListener('click', async () => {
        recordBtn.disabled = true;

        try {
            const response = await fetch('/record', { method: 'POST' });
            const data = await response.json();
            updateDisplay(data);
            await fetchTranscription();
        } catch (error) {
            console.error("Recording failed:", error);
        } finally {
            recordBtn.disabled = false;
        }
    });

    // Process uploaded file
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
            updateDisplay(data);
            await fetchTranscription();
        } catch (error) {
            console.error("Processing failed:", error);
        } finally {
            processBtn.disabled = false;
        }
    });
    // Update all visualizations
    function updateDisplay(data) {
        try {
            console.log("Raw data received:", data); // Debug complet
            
            // 1. Vérification des données reçues
            if (!data) {
                throw new Error("Aucune donnée reçue");
            }

            // 2. Mise à jour du lecteur audio
            if (data.audio && data.sample_rate) {
                console.log("Updating audio with sample rate:", data.sample_rate);
                updateAudioPlayer(data.audio, data.sample_rate);
            } else {
                console.warn("Données audio manquantes");
            }

            // 3. Mise à jour du spectrogramme
            if (data.spectrogram) {
                console.log("Updating spectrogram image");
                spectrogramImg.src = `data:image/png;base64,${data.spectrogram}`;
                spectrogramImg.onerror = () => {
                    console.error("Erreur de chargement du spectrogramme");
                    spectrogramImg.src = ''; // Réinitialiser si erreur
                };
            } else {
                console.warn("Spectrogramme manquant");
            }

            // 4. Mise à jour des MFCC
            if (data.mfcc && Array.isArray(data.mfcc)) {
                console.log("Updating MFCC visualization");
                drawMFCC(mfccCanvas, data.mfcc);
            } else {
                console.warn("Données MFCC manquantes ou invalides");
            }

        } catch (error) {
            console.error("Erreur dans updateDisplay:", error);
            // Vous pourriez aussi afficher un message à l'utilisateur ici
        }
    }

    // Play audio
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
        
        // Also update HTML5 player
        const wavBlob = audioToWavBlob(audioData, sampleRate);
        audioPlayer.src = URL.createObjectURL(wavBlob);
    }
    
    // Convert audio array to WAV blob
    function audioToWavBlob(audioData, sampleRate) {
        const buffer = new ArrayBuffer(44 + audioData.length * 2);
        const view = new DataView(buffer);
        
        // Write WAV header
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + audioData.length * 2, true);  // corrigé 32 → 36 pour taille correcte
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
        
        // Write audio samples
        const floatTo16Bit = (sample) => {
            sample = Math.max(-1, Math.min(1, sample));
            return sample < 0 ? sample * 32768 : sample * 32767;
        };
        
        for (let i = 0; i < audioData.length; i++) {
            view.setInt16(44 + i * 2, floatTo16Bit(audioData[i]), true);
        }
        
        return new Blob([view], { type: 'audio/wav' });
    }
    
    // Draw MFCC heatmap
    function drawMFCC(canvas, mfccData) {
        const container = canvas.parentElement;
        const width = container.clientWidth;
        const height = container.clientHeight;

        canvas.width = width;
        canvas.height = height;
        
        canvas.style.width = width + 'px';
        canvas.style.height = height + 'px';
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, width, height);
        const numFrames = mfccData.length;
        const numCoeffs = mfccData[0].length;
        
        const cellWidth = width / numFrames;
        const cellHeight = height / numCoeffs;
        
        // Find min/max for normalization
        let min = Infinity, max = -Infinity;
        mfccData.forEach(frame => {
            frame.forEach(val => {
                if (val < min) min = val;
                if (val > max) max = val;
            });
        });
        
        // Draw heatmap
        for (let frame = 0; frame < numFrames; frame++) {
            for (let coeff = 0; coeff < numCoeffs; coeff++) {
                const val = mfccData[frame][coeff];
                const norm = (val - min) / (max - min);
                
                // Create color gradient
                const hue = 240 * (1 - norm); // Blue to red
                ctx.fillStyle = `hsl(${hue}, 100%, 50%)`;
                
                ctx.fillRect(
                    frame * cellWidth,
                    (numCoeffs - coeff - 1) * cellHeight,
                    cellWidth,
                    cellHeight
                );
            }
        }
        
        // Add labels
        ctx.fillStyle = 'white';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        
        // Y-axis (MFCC coefficients)
        for (let coeff = 0; coeff < numCoeffs; coeff++) {
            ctx.fillText(
                `MFCC ${coeff}`,
                30,
                (numCoeffs - coeff - 0.5) * cellHeight
            );
        }
        
        // X-axis (time)
        for (let t = 0; t <= 30; t += 5) {
            const x = (t / 30) * canvas.width;
            ctx.fillText(
                `${t}s`,
                x,
                canvas.height - 5
            );
        }
    }
    // Get transcription from /transcribe/latest
async function fetchTranscription() {
    try {
        const response = await fetch('/transcribe/latest');
        if(!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        
        const transcriptionElement = document.getElementById('transcriptionText');
        if (data.transcription) {
            transcriptionElement.textContent = data.transcription;
        } else {
            transcriptionElement.textContent = "No transcription found.";
        }
    } catch (error) {
        console.error("Failed to fetch transcription:", error);
        document.getElementById('transcriptionText').textContent = "Error fetching transcription.";
    }
}

});
