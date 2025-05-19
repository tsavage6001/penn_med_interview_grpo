from dataclasses import dataclass
import torch
from typing import Optional

@dataclass
class GRPOConfig:
    model_name: str = "meta-llama/Llama-3.1-8B"
    learning_rate: float = 1.41e-5
    batch_size: int = 1
    max_length: int = 64
    num_turns: int = 4
    device: Optional[str] = None
    kl_coeff: float = 0.02
    branches: int = 2
    temperature: float = 0.7
    top_p: float = 0.9
    gradient_checkpointing: bool = True
    fp16: bool = False

    def __post_init__(self):
        """Validate configuration and set default device"""
        if self.device is None:
            # self.device = "cuda" # if torch.cuda.is_available() else "cpu"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.branches < 1:
            raise ValueError("branches must be at least 1")
        if not 0 < self.temperature <= 1.0:
            raise ValueError("temperature must be between 0 and 1")
        if not 0 < self.top_p <= 1.0:
            raise ValueError("top_p must be between 0 and 1")
