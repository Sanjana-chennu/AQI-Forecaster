import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch # <-- Import PyTorch
from sklearn.preprocessing import MinMaxScaler


# --- Add the project's root directory to the Python path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- IMPORTANT: Import the NEW PyTorch model and the preprocessing functions ---
from core.model import LSTMModel
from core.preprocess import load_and_clean_data, scale_and_create_sequences

def convert_pm25_to_aqi(pm25_value: float) -> int:
    """Converts a PM2.5 concentration value to the corresponding AQI value."""
    # (This function is correct and remains the same)
    if 0 <= pm25_value <= 12.0: return int(((50 - 0) / (12.0 - 0)) * (pm25_value - 0) + 0)
    elif 12.1 <= pm25_value <= 35.4: return int(((100 - 51) / (35.4 - 12.1)) * (pm25_value - 12.1) + 51)
    elif 35.5 <= pm25_value <= 55.4: return int(((150 - 101) / (55.4 - 35.5)) * (pm25_value - 35.5) + 101)
    elif 55.5 <= pm25_value <= 150.4: return int(((200 - 151) / (150.4 - 55.5)) * (pm25_value - 55.5) + 151)
    elif 150.5 <= pm25_value <= 250.4: return int(((300 - 201) / (250.4 - 150.5)) * (pm25_value - 150.5) + 201)
    else: return 500 # Simplified for brevity

# In bot/tools.py

def get_forecast(horizon_hours: int) -> dict:
    # ... (try/except block is the same) ...
 
    
    # --- THIS IS THE FIX ---
    # The LangChain agent passes the input as a string. We must convert it to an integer.
    try:
        horizon_hours = int(horizon_hours)
    except (ValueError, TypeError):
        # This will catch cases where the input is not a number (e.g., "twelve")
        return {"error": "Invalid input. Please provide a valid number of hours."}
    # -----------------------

    print(f"--- Running Multi-Output PyTorch forecast for {horizon_hours} hours ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    weights_path = os.path.join(project_root, 'saved_models', 'pytorch_multi_output_lstm.pth')

    if not os.path.exists(weights_path):
        return {"error": "Multi-output model weights not found. Please train the model."}

    # --- 1. Load Everything from the Checkpoint ---
    checkpoint = torch.load(weights_path, map_location=device,weights_only=False)
    model_config = checkpoint['model_config']
    df_columns = checkpoint['df_columns']
    target_columns = checkpoint['target_columns']
    
    # Recreate the scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.scale_ = checkpoint['scaler_state']
    scaler.min_ = checkpoint['scaler_min']

    # --- 2. Load the Model ---
    model = LSTMModel(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- 3. Get the Last Known Window of Data ---
    data_path = os.path.join(project_root, 'data', 'LSTM-Multivariate_pollution.csv')
    df = load_and_clean_data(data_path)
    
    last_window_unscaled = df.tail(48).values # Get the last 48 hours
    current_window_scaled = scaler.transform(last_window_unscaled)

    # --- 4. The New Multi-Output Recursive Prediction Loop ---
    future_predictions_scaled = []
    with torch.no_grad():
        for _ in range(horizon_hours):
            input_tensor = torch.from_numpy(current_window_scaled).float().unsqueeze(0).to(device)
            
            # Predict the next time step's target features (e.g., the 7 values)
            predicted_targets_scaled = model(input_tensor).cpu().numpy().flatten()
            future_predictions_scaled.append(predicted_targets_scaled)

            # --- Create the next input row ---
            # Start with a copy of the last known full row
            new_row_scaled = current_window_scaled[-1, :].copy()
            
            # Find the indices of the columns we need to update
            target_indices = [df_columns.index(col) for col in target_columns]
            
            # Replace the old values with our new predictions
            new_row_scaled[target_indices] = predicted_targets_scaled
            
            # Slide the window
            current_window_scaled = np.vstack([current_window_scaled[1:], new_row_scaled])

    # --- 5. Inverse Transform and Finalize ---
    future_predictions_scaled = np.array(future_predictions_scaled)
    
    # Create a dummy array to inverse transform
    dummy_pred = np.zeros((future_predictions_scaled.shape[0], len(df_columns)))
    dummy_pred[:, target_indices] = future_predictions_scaled
    predictions_actual = scaler.inverse_transform(dummy_pred)
    predictions_df = pd.DataFrame(predictions_actual, columns=df_columns)
    
    # Extract the final pollution and AQI values
    predictions_pm25 = predictions_df['pollution'].values
    predictions_aqi = [convert_pm25_to_aqi(p) for p in predictions_pm25]

    # ... (Plotting and return logic is the same as before) ...
    # (Just make sure the plot path is correct)
    plot_path = os.path.join(project_root, 'app', 'forecast_plot.png')
    # ... rest of plotting and returning ...
    # --- 6. Save Plot ---
    plot_path = os.path.join(project_root, 'app', 'forecast_plot.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.figure(figsize=(10, 6)); plt.plot(range(1, horizon_hours + 1), predictions_aqi, marker='o', linestyle='-');
    plt.title(f'AQI Forecast for the Next {horizon_hours} Hours (PyTorch Model)'); plt.xlabel('Hours from Now');
    plt.ylabel('Predicted AQI'); plt.grid(True); plt.savefig(plot_path); plt.close();
    print(f"Forecast plot saved to {plot_path}")

    # --- 7. Return the Results ---
    pm25_list = [round(p, 2) for p in predictions_pm25.flatten().tolist()]
    print(f"Forecast complete. PM2.5: {pm25_list}, AQI: {predictions_aqi}")
    
    return {
        "pm25_predictions": pm25_list,
        "aqi_predictions": predictions_aqi,
        "plot_path": plot_path
    }