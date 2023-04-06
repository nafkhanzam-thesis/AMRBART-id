from transformers import AutoModel

model = AutoModel.from_pretrained("models/mbart-large-50")

pytorch_total_params = sum(p.numel() for p in model.parameters())

print("pytorch_total_params:", pytorch_total_params)
