# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers[torch]",
#   "fire",
#   "torch",
#   "datasets",
#   "smolbox@git+https://github.com/attentionmech/smolbox",
# ]
# ///

import os

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

from smolbox.core.state_manager import AUTORESOLVE, resolve
from smolbox.core.base_tool import BaseTool



class ModelFineTuner(BaseTool):
    def __init__(
        self,
        model_path: str = AUTORESOLVE,
        dataset_path: str = AUTORESOLVE,
        output_model_path: str = AUTORESOLVE,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        max_seq_length: int = 512,
    ):
        """
        Fine-tune a model on a dataset.

        Args:
            model_path: Path to the model to fine-tune (local or HF hub)
            dataset_path: Path to the dataset (HF dataset name or local path)
            output_model_path: Where to save the fine-tuned model
            num_train_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
            max_seq_length: Maximum sequence length for tokenization
        """
        model_path = resolve("model_path", model_path)
        dataset_path = resolve("dataset_path", dataset_path)
        output_model_path = resolve("output_model_path", output_model_path, write=True)

        print(f"Model path: {model_path}")
        print(f"Dataset path: {dataset_path}")
        print(f"Output path: {output_model_path}")

        if os.path.exists(output_model_path) and os.listdir(output_model_path):
            print(f"WARNING: Output directory {output_model_path} is non empty.")

        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_path = output_model_path
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length

    def _prepare_dataset(self, tokenizer):
        """Load and preprocess the dataset."""
        print(f"Loading dataset from: {self.dataset_path}")
        dataset = load_dataset(self.dataset_path)

        # Assume a text field named 'text' exists; adjust if needed
        if "train" not in dataset:
            raise ValueError("Dataset must have a 'train' split")

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length,
            )

        print("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[
                col for col in dataset["train"].column_names if col != "text"
            ],
        )

        tokenized_dataset = tokenized_dataset.map(
            lambda x: {"labels": x["input_ids"]},
            batched=True,
        )

        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        
        return tokenized_dataset["train"]

    def run(self):
        """Fine-tune the model on the dataset."""
        print(f"Loading model from: {self.model_path}")
        model = AutoModelForCausalLM.from_pretrained(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Set padding token if not already set (common for GPT-style models)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        # Prepare dataset
        train_dataset = self._prepare_dataset(tokenizer)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_path,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            logging_dir=f"{self.output_path}/logs",
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            overwrite_output_dir=True,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        print("Starting fine-tuning...")
        trainer.train()

        print(f"Saving fine-tuned model to: {self.output_path}")
        trainer.save_model(self.output_path)
        tokenizer.save_pretrained(self.output_path)

        return f"Model fine-tuned and saved to {self.output_path}"


if __name__ == "__main__":
    fire.Fire(ModelFineTuner)
