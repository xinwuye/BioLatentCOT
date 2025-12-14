import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from smi_ted_light.load import load_smi_ted
from model import Qwen3MoleculeLLM, QueryAttentionProjector  # 替换成你模型文件路径

def load_trained_qwen3_model(model_dir, device="cuda"):
    """
    加载训练好的 Qwen3MoleculeLLM 模型，包括 tokenizer、LLM、projector。
    
    参数：
        model_dir: str
            保存模型的目录，里面应该包含：
            - ./llm/ 或 LLM 保存路径
            - projector.pt
            - full_model.pt 可选
        device: str
            "cuda" 或 "cpu"
    
    返回：
        model: Qwen3MoleculeLLM
    """

    # ---------------------------
    # 1. 加载 tokenizer + LLM
    # ---------------------------
    llm_path = os.path.join(model_dir, "qwen3_mol_sft_llm")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
    llm = AutoModelForCausalLM.from_pretrained(llm_path, local_files_only=True).to(device)

    # ---------------------------
    # 2. 初始化模型
    # ---------------------------
    model = Qwen3MoleculeLLM(qwen_model_name=llm_path).to(device)
    model.llm = llm
    model.tokenizer = tokenizer

    # ---------------------------
    # 3. 加载 projector
    # ---------------------------
    # projector_path = os.path.join(model_dir, "projector.pt")
    # if os.path.exists(projector_path):
    #     model.projector.load_state_dict(torch.load(projector_path, map_location=device))
    # else:
    #     print("[Warning] projector.pt not found, using randomly initialized projector.")

    # # ---------------------------
    # # 4. 可选：加载 full model state dict
    # # ---------------------------
    # full_model_path = os.path.join(model_dir, "qwen3_mol_sft_full.pt")
    # if os.path.exists(full_model_path):
    #     model.load_state_dict(torch.load(full_model_path, map_location=device))
    #     print("[Info] Full model state_dict loaded.")

    # ---------------------------
    # 5. 设置模型为 eval 模式（可训练可切换）
    # ---------------------------
    model.eval()

    return model, tokenizer


device = "cuda"
model_dir = "/zengdaojian/zhangjia/BioLatent/smi-ted/smi-ted/model_dir"  # 模型保存目录
model, tokenizer = load_trained_qwen3_model(model_dir, device=device)

# 推理示例
smiles_list = ["CCCCOC(C)C(=O)NCC(Cc1ccccc1)C(=O)[O-]."]
texts = ["Describe the functional groups of this molecule."]
enc = tokenizer(texts, return_tensors="pt", padding=True)
text_ids = enc["input_ids"].to(device)

with torch.no_grad():
    logits = model(smiles_list, text_ids)
    L_smiles = model.projector.num_queries + 2
    text_logits = logits[:, L_smiles:, :]
    pred_ids = text_logits.argmax(dim=-1)
    pred_text = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    print("Generated text:", pred_text)
