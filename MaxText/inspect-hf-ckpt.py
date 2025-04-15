import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def inspect_checkpoint(model_name: str, input_text: str, max_length: int = 50):
    """
    Inspects a Hugging Face checkpoint, prints model architecture details, and performs a simple generation test.

    Args:
        model_name (str): Hugging Face model name or local checkpoint path.
        input_text (str): Input text for the generation test.
        max_length (int): Maximum length of the generated text.
    """
    # Load model and tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    # Print model architecture details
    print("Model Architecture Details:")
    print(f"Model Type: {model.config.model_type}")
    print(f"Number of Layers: {model.config.num_hidden_layers}")
    print(f"Hidden Size: {model.config.hidden_size}")
    print(f"Number of Attention Heads: {model.config.num_attention_heads}")
    print(f"Vocab Size: {model.config.vocab_size}")
    print(f"Max Position Embeddings: {model.config.max_position_embeddings}")

    total_params = sum(p.numel() for p in model.parameters())
    # Count the number of 0s in the model
    zero_params = sum(torch.sum(p == 0).item() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    print(f"Zero Parameters: {zero_params} ({zero_params / total_params * 100:.2f}%)")

    print(model)

    # Perform a simple generation test
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"], 
            max_length=max_length, 
            num_return_sequences=1, 
            do_sample=False, 
            temperature=0.7
        )

    # Decode and print generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nGenerated Text:")
    print(generated_text)

# Example usage
if __name__ == "__main__":
    # Take model name from first argument
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    input_text = "I love to"
    inspect_checkpoint(model_name, input_text)
