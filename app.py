from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Inisialisasi aplikasi FastAPI
app = FastAPI(title="Prediksi Gangguan Tidur")

# Memuat model dan scaler yang sudah disimpan
try:
    with open("dtc_Model.pkl", "rb") as f:
        model_data = pickle.load(f)
        dtc_model = model_data['model']  # Ambil model dari dictionary
        scaler = model_data['scaler']  # Ambil scaler dari dictionary

except FileNotFoundError as e:
    raise HTTPException(status_code=500, detail=f"Error loading model or scaler: {str(e)}")

# Mendefinisikan struktur data input menggunakan Pydantic
class SleepData(BaseModel):
    gender: int  # '1' untuk Male atau '0' untuk Female
    age: float
    occupation: int  # Angka untuk pekerjaan
    sleep_duration: float
    quality_of_sleep: float
    physical_activity_level: float
    stress_level: float
    bmi_category: int  # 0 untuk 'Normal', 1 untuk 'Overweight', 2 untuk 'Obese'
    heart_rate: float

    class Config:
        allow_population_by_field_name = True

# Fungsi untuk preprocessing input
def preprocess_input(data: SleepData):
    # Buat DataFrame dari input
    input_data = pd.DataFrame([{
        "gender": data.gender,
        "age": data.age,
        "occupation": data.occupation,
        "sleep_duration": data.sleep_duration,
        "quality_of_sleep": data.quality_of_sleep,
        "physical_activity_level": data.physical_activity_level,
        "stress_level": data.stress_level,
        "bmi_category": data.bmi_category,
        "heart_rate": data.heart_rate
    }])

    # Menyelaraskan urutan kolom sesuai dengan yang digunakan saat pelatihan model
    expected_columns = ['gender', 'age', 'occupation', 'sleep_duration', 'quality_of_sleep', 
                        'physical_activity_level', 'stress_level', 'bmi_category', 'heart_rate']

    # Menyusun ulang kolom jika ada ketidaksesuaian
    input_data = input_data[expected_columns]

    # Normalisasi data dengan scaler
    input_data_normalized = scaler.transform(input_data)
    return input_data_normalized

# Endpoint untuk prediksi
@app.post("/predict/")
def predict_sleep_disorder(data: SleepData):
    try:
        # Menyiapkan input data sesuai format yang diperlukan oleh model
        processed_data = preprocess_input(data)

        # Melakukan prediksi menggunakan model Decision Tree
        prediction = dtc_model.predict(processed_data)

        # Menangani hasil prediksi dan memberikan hasil deskriptif
        if prediction[0] == 1:
            prediction_result = "Gangguan Tidur"
        else:
            prediction_result = "Tidak Ada Gangguan Tidur"

        # Mengembalikan hasil prediksi dengan label deskriptif
        return {"prediction": prediction_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")