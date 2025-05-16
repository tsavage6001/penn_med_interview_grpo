#!/usr/bin/env python3
from transformers import AutoTokenizer
import pandas as pd
from grpo_trainer import (
    GRPOConfig,
    GRPOTrainer,
    setup_environment,
    print_heading
)

def main():
    setup_environment()
    print_heading("GRPO Medical Dialogue Trainer")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/phi-1_5",
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Configuration
    config = GRPOConfig(
        branches=2,
        num_turns=4,
        max_length=128,
        fp16=False
    )


    # Medical cases DataFrame
    medical_data = pd.DataFrame({
        'case': [
            "I've had persistent headaches and nausea for two weeks",
            "I experience chest pain when climbing stairs",
            "I'm unusually thirsty",
            "I've been feeling extremely tired for months",
            "I have sharp abdominal pain after eating"
        ],
        'details': [
            "Its worse when I lie down.  I have had diarrhea.  I have not vomited",
            "It feels like pressure.  It is worse when I am active.  It goes away with rest",
            "I've been urinating a lot and I'm always thirsty.  I have lost weight",
            "My hair is falling out.  I have dry skin.  I feel cold all the time",
            "I take ibuprofen for the pain.  I have been nauseous.  I have lost weight"
        ],
        'diagnosis': [
            "Elevated intracranial pressure",
            "Stable Angina",
            "diabetes mellitus",
            "hypothyroidism",
            "peptic ulcer"
        ]
    })

    # Create trainer
    trainer = GRPOTrainer(tokenizer, config, medical_data)

    # Training loop
    for _, row in medical_data.iterrows():
        print_heading(f"Processing case: {row['case'][:50]}...")
        trainer.train_on_conversation_tree(row['case'])

if __name__ == "__main__":
    main()