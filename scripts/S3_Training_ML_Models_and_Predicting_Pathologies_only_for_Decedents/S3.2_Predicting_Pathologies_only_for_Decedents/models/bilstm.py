import torch.nn as nn
import torch
class BiLSTM(nn.Module):
    """
    An LSTM-based model for sequence prediction tasks with Batch Normalization.
    Now supports bidirectional LSTM.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.5, bidirectional=True):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units in each LSTM layer.
            output_dim (int): Number of output features.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
            bidirectional (bool): Whether to use bidirectional LSTM.
        """
        super(BiLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            dropout=dropout if num_layers > 1 else 0.0, 
            batch_first=True, 
            bidirectional=bidirectional
        )
        
        # Adjust BatchNorm and FC input dimensions based on bidirectionality
        self.batch_norm = nn.BatchNorm1d(hidden_dim * self.num_directions)
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

    def forward(self, x):
        # LSTM returns: (output, (hidden, cell))
        _, (hidden, _) = self.lstm(x)  
        # If using a multi-layer bidirectional LSTM, the final forward hidden state is stored at hidden[-2], and the final backward hidden state is stored at hidden[-1].
        if self.bidirectional:
            forward_hidden = hidden[-2]  # shape: (batch_size, hidden_dim)
            backward_hidden = hidden[-1]  # shape: (batch_size, hidden_dim)
            out = torch.cat((forward_hidden, backward_hidden), dim=1)  # shape: (batch_size, hidden_dim * 2)
        else:
            out = hidden[-1]  # shape: (batch_size, hidden_dim)
            
        out = self.batch_norm(out)
        out = self.fc(out)
        return out