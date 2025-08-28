import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_prob=0.2):
        """
        Initializes the model layers.
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 1. The core LSTM Layer (Stacked with Dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout_prob)
        
        # 2. The Final Fully-Connected Layer
        self.fc = nn.Linear(hidden_size, output_size)

        # 3. The Activation Function Layer (for stability)
        # This will prevent the model from predicting negative pollution values.
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Pass input through the LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        
        # Pass the output of the last time step through the fully connected layer
        out = self.fc(out[:, -1, :])
        
        # --- THIS IS THE CRITICAL FIX ---
        # Apply the ReLU activation function to the final output.
        # This ensures the model's output can never be negative.
        out = self.relu(out)
        # -------------------------------
        
        return out