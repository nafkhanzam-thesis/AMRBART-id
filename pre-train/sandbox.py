import json
from model_interface.tokenization_bart import AMRBartTokenizer


data = []
with open('../datasets/amrbart-test/pretrain.jsonl') as f:
    data.extend([json.loads(line)['amr'] for line in f])

tokenizer = AMRBartTokenizer.from_pretrained("../models/tiny-mbart")

# print(tokenizer.INIT + '<pointe:123>' in tokenizer.vocab)

data0 = data[0].split()

res = tokenizer.tokenize_amr(data0)
print(data0)
print(res)
