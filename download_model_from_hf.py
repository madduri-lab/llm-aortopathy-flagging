# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")

# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")