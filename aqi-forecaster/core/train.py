import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

from preprocess import load_and_clean_data, scale_and_create_sequences
from model import LSTMModel

if __name__ == "__main__":
    
    device = "cpu"
    print(f"Using device: {device}")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(project_root, 'data', 'LSTM-Multivariate_pollution.csv')
    df = load_and_clean_data(file_path)
    X_train, y_train, X_test, y_test, scaler, df_columns, target_columns = scale_and_create_sequences(df)

    #Create DataLoaders
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    batch_size = 64
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    input_size = X_train.shape[2]
    hidden_size = 80  #I reduced hidden size to 80 from 128 to prevent overfitting
    output_size = y_train.shape[1]
    num_layers = 2
    dropout_prob = 0.2

    model = LSTMModel(input_size, hidden_size, output_size, num_layers, dropout_prob).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    #The Training Loop 
    epochs = 50
    print("\n--- Starting Final Multi-Output Model Training ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.8f}")

    print("\n--- Training Complete ---")

    # --- 5. Save the Model ---
    save_dir = os.path.join(project_root, 'saved_models')
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'pytorch_multi_output_lstm.pth')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': input_size, 'hidden_size': hidden_size, 'output_size': output_size,
            'num_layers': num_layers, 'dropout_prob': dropout_prob
        },
        'scaler_state': scaler.scale_, 'scaler_min': scaler.min_,
        'df_columns': df_columns.tolist(), 'target_columns': target_columns
    }, save_path)
    print(f"Final model saved successfully to '{save_path}'")

    # --- 6. Full Model Evaluation on the Test Set ---
    print("\n--- Evaluating Final Model on Full Test Set ---")
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy())
            
    predictions_scaled = np.concatenate(all_predictions, axis=0)
    
    # Inverse transform
    dummy_pred = np.zeros((predictions_scaled.shape[0], len(df_columns)))
    target_indices = [df_columns.get_loc(col) for col in target_columns]
    dummy_pred[:, target_indices] = predictions_scaled
    predictions_actual = scaler.inverse_transform(dummy_pred)
    predictions_df = pd.DataFrame(predictions_actual, columns=df_columns)

    dummy_true = np.zeros((y_test.shape[0], len(df_columns)))
    dummy_true[:, target_indices] = y_test
    y_test_actual = scaler.inverse_transform(dummy_true)
    y_test_df = pd.DataFrame(y_test_actual, columns=df_columns)

    # Calculate RMSE for each target variable
    print("\n--- RMSE for each predicted feature ---")
    for col in target_columns:
        rmse = np.sqrt(np.mean((predictions_df[col] - y_test_df[col])**2))
        print(f"  {col}: {rmse:.4f}")
        
    # Visualize Pollution Prediction
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_df['pollution'], color='blue', label='Actual Pollution')
    plt.plot(predictions_df['pollution'], color='red', linestyle='--', label='Predicted Pollution')
    plt.title('Pollution Prediction vs Actual (Final Multi-Output Model)')
    plt.xlabel('Time Steps (Hours)'); plt.ylabel('PM2.5 Concentration')
    plt.legend()
    evaluation_plot_path = os.path.join(project_root, 'evaluation_results_final.png')
    plt.savefig(evaluation_plot_path)
    print(f"\nFinal evaluation plot saved to: {evaluation_plot_path}")