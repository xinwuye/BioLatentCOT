from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append('/zengdaojian/zhangjia/BioLatent/smi-ted/smi-ted/inference')
from smi_ted_light.load import load_smi_ted
import pandas as pd
import numpy as np
import torch

# Chemistry
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.DataStructs import TanimotoSimilarity

# function to canonicalize SMILES
def normalize_smiles(smi, canonical=True, isomeric=False):
    try:
        normalized = Chem.MolToSmiles(
            Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
        )
    except:
        normalized = None
    return normalized

# function to calculate pairwise Tanimoto similarity
def calculate_tanimoto_similarities(fps1, fps2):
    similarities = []
    for i in range(len(fps1)):
            sim = TanimotoSimilarity(fps1[i], fps2[i])
            similarities.append(sim)
    return similarities


model_smi_ted = load_smi_ted(
    folder='/zengdaojian/zhangjia/BioLatent/smi-ted/smi-ted/inference/smi_ted_light',
    ckpt_filename='/zengdaojian/zhangjia/BioLatent/smi-ted/smi-ted-Light_40.pt'
)

df_moses = pd.read_csv("/zengdaojian/zhangjia/BioLatent/smi-ted/smi-ted/notebooks/data/moses_test.csv", nrows=1000)

with torch.no_grad():
    # print(type(df_moses['SMILES']))
    # encode_embeddings = model_smi_ted.encode(df_moses['SMILES'], return_torch=True)
    encode_embeddings = model_smi_ted.encode(["c1ccccc1"], return_torch=True)
    
print(encode_embeddings.shape)
import torch
import torch.nn as nn

# 输入张量
# input_tensor = torch.randn(1, 202, 768)
# print(f"输入形状: {input_tensor.shape}")

# 将序列维度展平，然后用线性层映射
batch_size, seq_len, feature_dim = encode_embeddings.shape
reshaped = encode_embeddings.view(batch_size, -1)  # [1, 202*768] = [1, 155136]

# 定义线性层直接映射到目标尺寸
linear_layer = nn.Linear(seq_len * feature_dim, 2560)  # 155136 -> 2560
output = linear_layer(reshaped).to(torch.bfloat16)  # [1, 2560]
print(type(output))







model_name = "/zengdaojian/zhangjia/BioLatent/Qwen4B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 重塑为所需的输出形状
output = output.view(1, 1, 2560).to(model.device)  # [1, 1, 2560]
print(f"输出形状: {output.shape}")

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

model_name = "/zengdaojian/zhangjia/BioLatent/Qwen4B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)



print(f"输入 token 数量: {model_inputs['input_ids'].shape[1]}")

# 获取原始的 token embeddings
embedding_layer = model.get_input_embeddings()
original_embeddings = embedding_layer(model_inputs['input_ids'])  # [1, seq_len, embed_dim]
print(f"原始 embeddings 形状: {original_embeddings.shape}")

# # 创建你要拼接的向量（假设是 [1, 1, embed_dim] 形状）
# additional_vector = torch.randn(1, 1, original_embeddings.shape[-1]).to(original_embeddings.device)
# print(f"额外向量形状: {additional_vector.shape}")

# 在序列维度上拼接 [1, seq_len, embed_dim] + [1, 1, embed_dim] -> [1, seq_len+1, embed_dim]
combined_embeddings = torch.cat([original_embeddings, output], dim=1)
print(f"拼接后 embeddings 形状: {combined_embeddings.shape}")

# 使用拼接后的 embeddings 作为输入进行前向传播
# with torch.no_grad():
#     outputs = model(inputs_embeds=combined_embeddings)
    
# print(f"输出 logits 形状: {outputs.logits.shape}")


# ✅ 正确：使用 generate 而不是直接调用 model
with torch.no_grad():
    generated_ids = model.generate(
        inputs_embeds=combined_embeddings,
        max_new_tokens=512,
        do_sample=False,  # or True with temperature/top_p
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

# 解码生成的内容（只解码新生成的部分）
prompt_token_len = model_inputs['input_ids'].shape[1]  # 原始 prompt 的 token 数
# 注意：因为你额外加了 1 个 token（output 的 seq_len=1），所以总输入长度是 prompt_token_len + 1
# 但你真正想跳过的“非生成部分”是 combined_embeddings 的整个长度！
input_seq_len = combined_embeddings.shape[1]  # 这才是实际输入的长度

generated_text = tokenizer.decode(
    generated_ids[0][input_seq_len:],  # 跳过所有输入部分
    skip_special_tokens=True
).strip()

print("Generated text:", generated_text)



# generated_ids = model.generate(
#     combined_embeddings,
#     max_new_tokens=32768
# )
# output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# # parsing thinking content
# try:
#     # rindex finding 151668 (</think>)
#     index = len(output_ids) - output_ids[::-1].index(151668)
# except ValueError:
#     index = 0

# thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
# content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# print("thinking content:", thinking_content)
# print("content:", content)

