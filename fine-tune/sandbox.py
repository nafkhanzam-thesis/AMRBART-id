import json
from model_interface.tokenization_bart import AMRBartTokenizer


data = ["( <pointer:0> take-10 :ARG0 ( <pointer:1> it ) :ARG1 ( <pointer:2> long-03 :polarity - :ARG1 <pointer:1> ) )"]
# with open('../datasets/amrbart-test/pretrain.jsonl') as f:
#     data.extend([json.loads(line)['amr'] for line in f])

tokenizer = AMRBartTokenizer.from_pretrained("../models/mbart-en-id-smaller")

# print(tokenizer.INIT + '<pointe:123>' in tokenizer.vocab)

data0 = data[0].split()

res = tokenizer.tokenize_amr(data0)
print(data0)
print(res)
back = tokenizer.decode_amr(res)
print(back)
