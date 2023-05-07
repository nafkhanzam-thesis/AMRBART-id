from model_interface.tokenization_mbart50 import MBart50Tokenizer
from model_interface.modeling_bart import MBartForConditionalGeneration

model_path = "models/mbart-en-id-smaller"
tokenizer = MBart50Tokenizer.from_pretrained(model_path)
model = MBartForConditionalGeneration.from_pretrained(model_path)

model.resize_token_embeddings(len(tokenizer))

print(model)
