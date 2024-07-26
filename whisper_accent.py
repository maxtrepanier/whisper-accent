""" Implements the accent classifier model using transfer learning on whisper.
"""
__author__ = "Maxime Trepanier"

import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration

# class generated with Claude
class AccentClassifier(nn.Module):
    """ Accent classifier layers
    """
    def __init__(self, input_dim=768, num_classes=2, dropout_rate=0.5):
        """
        Args:
         - input_dim: number of audio channels (768 for whisper-small)
         - num_classes: nb of output classes
         - dropout_rate: dropout layer rate for learning
        """
        super(AccentClassifier, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """ Runs x forward through the model.
        Args:
         - x: a tensor of shape (batch_size, 1500, input_dim)
        Returns:
         - logit for classifier, shape (batch_size, num_classes)
        """
        # x shape: (batch_size, 1500, 768) (output of Whisper encoder)

        # Permute dimensions for pooling
        x = x.permute(0, 2, 1)  # shape: (batch_size, 768, 1500)

        # Apply average pooling
        x = self.avg_pool(x)  # shape: (batch_size, 768, 1)

        # Flatten
        x = x.view(x.size(0), -1)  # shape: (batch_size, 768)

        # Apply dropout
        x = self.dropout(x)

        # Fully connected layer
        x = self.fc(x)

        return x

# Example usage:
# whisper_output = torch.randn(32, 1500, 768)  # Example batch of Whisper encoder outputs
# model = AccentClassifier()
# output = model(whisper_output)
# print(output.shape)  # Should be torch.Size([32, 2])

class AccentModel(nn.Module):
    def __init__(self, num_classes: int = 2, use_encoder: bool = True, model_id: str = "openai/whisper-small"):
        super(AccentModel, self).__init__()

        self.use_encoder = use_encoder

        # config
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # load model
        self.whisper_encoder = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
           ).model.encoder

        # Freeze Whisper encoder weights
        for param in self.whisper_encoder.parameters():
            param.requires_grad = False

        self.accent_classifier = AccentClassifier(num_classes=num_classes)

    def encode(self, x):
        with torch.no_grad():
            return self.whisper_encoder(x).last_hidden_state

    def forward(self, x):
        if self.use_encoder:
            encoder_output = self.encode(x)
        else:
            encoder_output = x
        return self.accent_classifier(encoder_output)

# Example usage:
# model = FullAccentModel()
# input_features = torch.randn(32, 80, 3000)  # Example input to Whisper
# output = model(input_features)
# print(output.shape)  # Should be torch.Size([32, 2])
