import torch
from torchview import draw_graph
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

model_path = "models/mbart-en-id-smaller"
model = MBartForConditionalGeneration.from_pretrained(model_path)
tokenizer = MBart50Tokenizer.from_pretrained(model_path)
# device='meta' -> no memory is consumed for visualization
input_data = tokenizer(
    "aku adalah manusia",
    max_length=512,
    padding=False,
    truncation=True,
)['input_ids']
input_data = torch.tensor([input_data])
print(input_data.shape)
model_graph = draw_graph(
    model,
    # input_size=(8,),
    input_data=input_data,
    expand_nested=True,
    roll=True,
    device='meta',
)
# model_graph.visual_graph
