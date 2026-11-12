"""
model.py

What is done here:
    - Model Architecture
        - Create Instance of class BiLSTMModel (Autoencoder Model):
        - Encoder (BiLSTM):
            - Input: Sequence of 6 features per timestep
            - Output: Final Forward Hidden State h_T^(f) and
                      final backward hidden state h_0^(b)
            - Size: 64(forward) + 64(backward) = 128-dimensional encoder representation

        - Compressor (Bottleneck AKA Latent Code (z)):
            - Concatenate [h_T^(f), h_0^(b)]
            - Compress 128 -> 64
            - This 64-dimensional latent vector z captures user's behavioral patterns

        - Decoder(LSTM):
            - Repeat latent vector z across seq_len timesteps
            - Decode back into a sequence of hidden states

        - Output layer:
            - Map decoder hidden state (64 dim) to original 6 features per timestep

        Final output: Reconstruction of the original sequence: Shape [Batch, seq_len, 6]
"""
import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim, num_layers):
        """
        Args:
            seq_len (int): Number of timesteps (e.g., 50)
            n_features (int): Number of input features per timestep. (6)
            embedding_dim (int): Hidden size inside LSTM. (64)
            num_layers (int): Number of LSTM layers. (1)
        """
        super(BiLSTMModel, self).__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Encoder (BiLSTM):
        """
        Forward Hidden States: h_t^(f)
        Backward Hidden States: h_t^(b)
        
        Use Final Hidden States from both:
            - h_T(f)  : Last Forward Hidden
            - h_0^(b) : Last Backward Hidden
        """
        self.encoder = nn.LSTM(
            input_size=n_features,     # Input features (6) -> hidden states (64 numbers)
            hidden_size=embedding_dim, # Output 64 numbers each forward & backward
            num_layers=num_layers,
            bidirectional=True,        # Bidirectional=True means it reads the sequence FORWARDS & BACKWARDS
            batch_first=True
        )

        # Compressor (Bottleneck): Important for learning useful representation:
        self.encoder_linear = nn.Linear(embedding_dim * 2, # 128 (F + B) -> Compressed Size 64 = Latent Code z
                                        embedding_dim)

        # Decoder (LSTM): Takes Latent Code (z) and repeats it across all timesteps
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        # Output layer: Turn Decoder's learned representation into the 6 original features
        self.output_layer = nn.Linear(embedding_dim, n_features)

    def forward(self, x):
        # Encoding:
        """
        output (ignored)
        (h_n, c_n):  h_n shape = [num_layers * num_directions, batch_size, hidden_size]
        BiLSTM with 1 layer:
            h_n[0] = last forward hidden state h_T^(f)
            h_n[1] = last backward hidden state h_0^(b)

            h_n = hidden
        """
        (_,                     # Ignore the first output at each time step, but grab the final hidden state
         (hidden, _)) \
            = self.encoder(x)   # Pass the entire sequence through the BiLSTM encoder

        # Concatenate forward and backward (the 2 layers in hidden state) together -> Shape: [Batch, 128]
        hidden_concat = torch.cat((hidden[-2,:,:],    # Forward Hidden[B, 64]
                                hidden[-1,:,:]), dim=1)       # Backward  Hidden[B, 64]

        # Compress to our bottleneck: Shape -> [Batch, 64]
        z = self.encoder_linear(hidden_concat)          # This is our latent code

        # Decoding: Shape: [B, 50, 64]
        z_repeated = (z.unsqueeze(1)                    # Unsqueeze: Add time-step dimension -> [B, 1, 64]
                         .repeat(1, x.size(1), 1))      # Repeat: [B, 50, 64]
        decoded, _ = self.decoder(z_repeated)           # Decode full sequence

        # Map back to the original 6 features
        reconstruction = self.output_layer(decoded)

        return reconstruction
