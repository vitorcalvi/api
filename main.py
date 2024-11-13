from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import librosa
import numpy as np
import tempfile
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='librosa')

app = FastAPI()

def extract_audio_features(audio_file_path):
    # Load the audio file and extract features
    y, sr = librosa.load(audio_file_path, sr=None)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=75, fmax=600)
    f0 = f0[~np.isnan(f0)]
    energy = librosa.feature.rms(y=y)[0]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    speech_rate = tempo / 60
    return f0, energy, speech_rate, mfccs, y, sr

def analyze_voice_stress(audio_file_path):
    f0, energy, speech_rate, mfccs, y, sr = extract_audio_features(audio_file_path)
    mean_f0 = np.mean(f0)
    std_f0 = np.std(f0)
    mean_energy = np.mean(energy)
    std_energy = np.std(energy)
    gender = 'male' if mean_f0 < 165 else 'female'
    norm_mean_f0 = 110 if gender == 'male' else 220
    norm_std_f0 = 20
    norm_mean_energy = 0.02
    norm_std_energy = 0.005
    norm_speech_rate = 4.4
    norm_std_speech_rate = 0.5
    z_f0 = (mean_f0 - norm_mean_f0) / norm_std_f0
    z_energy = (mean_energy - norm_mean_energy) / norm_std_energy
    z_speech_rate = (speech_rate - norm_speech_rate) / norm_std_speech_rate
    stress_score = (0.4 * z_f0) + (0.4 * z_speech_rate) + (0.2 * z_energy)
    stress_level = float(1 / (1 + np.exp(-stress_score)) * 100)
    categories = ["Very Low Stress", "Low Stress", "Moderate Stress", "High Stress", "Very High Stress"]
    category_idx = min(int(stress_level / 20), 4)
    stress_category = categories[category_idx]
    return {"stress_level": stress_level, "category": stress_category, "gender": gender}

def analyze_text_stress(text: str):
    stress_keywords = ["anxious", "nervous", "stress", "panic", "tense"]
    stress_score = sum([1 for word in stress_keywords if word in text.lower()])
    stress_level = min(stress_score * 20, 100)
    categories = ["Very Low Stress", "Low Stress", "Moderate Stress", "High Stress", "Very High Stress"]
    category_idx = min(int(stress_level / 20), 4)
    stress_category = categories[category_idx]
    return {"stress_level": stress_level, "category": stress_category}

class StressResponse(BaseModel):
    stress_level: float
    category: str
    gender: str = None  # Optional, only for audio analysis

@app.post("/analyze-stress/", response_model=StressResponse)
async def analyze_stress(
    file: UploadFile = File(None), 
    file_path: str = Form(None),
    text: str = Form(None)
):
    if file is None and file_path is None and text is None:
        raise HTTPException(status_code=400, detail="Either a file, file path, or text input is required.")
    
    # Handle audio file analysis
    if file or file_path:
        if file:
            if not file.filename.endswith(".opus"):
                raise HTTPException(status_code=400, detail="Only .opus files are supported.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".opus") as temp_file:
                temp_file.write(await file.read())
                temp_file_path = temp_file.name
        else:
            if not file_path.endswith(".opus"):
                raise HTTPException(status_code=400, detail="Only .opus files are supported.")
            if not os.path.exists(file_path):
                raise HTTPException(status_code=400, detail="File path does not exist.")
            temp_file_path = file_path

        try:
            result = analyze_voice_stress(temp_file_path)
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if file:
                os.remove(temp_file_path)

    # Handle text analysis
    elif text:
        result = analyze_text_stress(text)
        return JSONResponse(content=result)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))  # Use the PORT environment variable for Render compatibility
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)