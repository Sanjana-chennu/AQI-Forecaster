# AQI-Forecaster: A Full-Stack Conversational AI for Time Series Forecasting

This project is a complete, end-to-end AI application that provides real-time air quality (AQI) forecasts through a natural language chat interface. It combines a state-of-the-art deep learning model, a conversational AI agent, and a modern full-stack web architecture.

<img width="1094" height="471" alt="image" src="https://github.com/user-attachments/assets/b4b90be2-5c30-482d-8223-6110a621a029" />

*(This plot shows the final LSTM model's predictions (red) against the actual pollution values (blue) on the unseen test set, demonstrating its ability to capture complex daily cycles.)*

---

## 🚀 Key Features

-   **High-Performance Time Series Model:** A multi-output, **Stacked LSTM** model built in PyTorch, achieving a strong **Root Mean Squared Error (RMSE)** on the pollution prediction task.
-   **Dynamic Forecasting:** The model performs a self-consistent simulation by forecasting 7 key environmental variables simultaneously, leading to realistic daily patterns.
-   **Conversational AI Agent:** A **LangChain agent** powered by Google's Gemini LLM understands user requests and intelligently calls the forecasting tool.
-   **Professional Full-Stack Architecture:** A decoupled system with a **Python/FastAPI backend** serving the AI engine and a **JavaScript/React frontend** providing the user interface.
-   **Advanced MLOps Concepts:** Includes implemented solutions for **online learning (fine-tuning)** to combat concept drift and **pseudo-labeling** for semi-supervised learning.

---

## 🛠️ The Technical Journey: Iterative Model Development

The final model was the result of a rigorous, multi-stage experimental process designed to find the optimal architecture for this specific dataset.

1.  **Baseline (Single-Output LSTM):** The project began with a simple single-output model. This revealed a key flaw: the inability to generate dynamic long-term forecasts, resulting in unrealistic "flat-line" predictions. This experiment was crucial as it proved a more sophisticated approach was necessary.

2.  **System Upgrade (Multi-Output LSTM):** To solve the "flat-line" problem, the architecture was re-engineered into a **multi-output system**. The core RNN chosen for this was the classic and powerful **LSTM (Long Short-Term Memory)**. The final production model is a **2-layer Stacked LSTM with Dropout**, which is highly effective at capturing long-term dependencies in the data.

3.  **Stability and Optimization:** During testing, the powerful LSTM model showed signs of instability (predicting negative values). An architectural fix was implemented by adding a **ReLU  activation function** to the final layer, ensuring all predictions are physically plausible and making the model robust.
THIS DOC FOLLOWS THE JOURNEY OF THE TRAINING THE MODEL-
https://docs.google.com/document/d/1urjKmjBLq7omXj52_XWXqU2dk04EKbJkPsQLS2LxWfE/edit?usp=sharing
THE WEB-APPLICATION
1.  **Frontend (`aqi-forecaster-ui`):** A **React** application that provides a polished chat interface. Its only job is to send user messages to the backend and display the responses.
2.  **Backend (`aqi-forecaster`):** A **FastAPI** server that exposes a single `/chat` endpoint. It receives requests from the frontend and passes them to the LangChain agent.
3.  **AI Engine (`core` & `bot` modules):** The LangChain agent understands the user's intent. If a forecast is requested, it calls the `get_forecast` tool, which loads the trained **PyTorch Stacked LSTM model** and executes the prediction.

---

## 📦 How to Run This Project

This project consists of two separate applications that must be run simultaneously.

### 1. Backend (FastAPI Server)

-   Navigate to the `aqi-forecaster` directory.
-   Create a `.env` file and add your `GOOGLE_API_KEY`.
-   Install dependencies: `pip install -r requirements.txt`
-   Run the server: `uvicorn main:app --reload`
-   The backend will be live at `http://12.0.0.1:8000`.

### 2. Frontend (React App)

-   Navigate to the `aqi-forecaster-ui` directory.
-   Install dependencies: `npm install`
-   Run the development server: `npm start`
-   Open your browser to `http://localhost:3000`.

---
*This project was developed as a submission for the college ML club selection process.*
