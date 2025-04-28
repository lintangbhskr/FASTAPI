from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Memuat model dan scaler
model = joblib.load('food_efficiency_model.pkl')
scaler = joblib.load('scaler_food_efficiency.pkl')

# Pydantic model untuk validasi input
class InputData(BaseModel):
    Quantity_of_Food: float
    Wastage_Food_Amount: float
    Number_of_Guests: float
    Event_Type_Encoded: float

@app.get("/")
def read_root():
    return {"message": "Selamat datang di API prediksi efisiensi makanan"}

@app.post("/predict")
def predict(data: InputData):
    try:
        # Konversi input data menjadi array
        input_data = np.array([[data.Quantity_of_Food, 
                                data.Wastage_Food_Amount, 
                                data.Number_of_Guests, 
                                data.Event_Type_Encoded]])

        # Melakukan scaling pada data input
        scaled_data = scaler.transform(input_data)

        # Prediksi dengan model
        prediction = model.predict(scaled_data)

        return {"prediction": prediction[0]}

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
