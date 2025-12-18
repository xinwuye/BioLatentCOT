import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import sys
sys.path.append('/zengdaojian/zhangjia/BioLatent/smi-ted/smi-ted/inference')
from smi_ted_light.load import load_smi_ted
import torch.nn.functional as F
from transformers.generation.utils import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional

import torch
from typing import List, Optional


# ============================
# 1. 投影器：将分子特征映射到LLM空间
# ============================
class QueryAttentionProjector(nn.Module):
    def __init__(self, input_dim=768, num_queries=4, output_dim=2560, num_heads=8):
        """
        简化版本的查询注意力投影器（包含必要的归一化）
        """
        super().__init__()
        self.num_queries = num_queries
        
        # 输入归一化
        self.input_norm = nn.LayerNorm(input_dim)
        
        # 可学习的查询向量
        self.query = nn.Parameter(torch.randn(1, num_queries, input_dim) * 0.02)
        
        # 查询向量归一化
        self.query_norm = nn.LayerNorm(input_dim)
        
        # 多头注意力
        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # 注意力后归一化
        self.post_attn_norm = nn.LayerNorm(input_dim)
        
        # 投影层
        self.proj = nn.Linear(input_dim, output_dim)
        
        # 投影前归一化
        self.pre_proj_norm = nn.LayerNorm(input_dim)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.query, gain=1.0)
        nn.init.xavier_uniform_(self.proj.weight, gain=1.0)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
    
    def forward(self, x):
        B = x.size(0)
        
        # 输入归一化
        x_norm = self.input_norm(x)
        
        # 准备查询向量
        q = self.query.expand(B, -1, -1)
        q = self.query_norm(q)
        
        # 注意力计算
        attn_out, _ = self.attn(q, x_norm, x_norm)
        
        # 残差连接 + 归一化
        residual = q + attn_out
        residual = self.post_attn_norm(residual)
        
        # 投影前归一化
        proj_input = self.pre_proj_norm(residual)
        
        # 最终投影
        out = self.proj(proj_input)
        
        return out


# ============================
# 2. 多模态融合模型 (兼容trl的SFTTrainer)
# ============================
class Qwen3MoleculeLLM(PreTrainedModel):
    def __init__(self, 
                 qwen_model_name="/zengdaojian/zhangjia/BioLatent/Qwen4B",
                 d_mol=202*768):  # d_mol参数保留用于兼容性，实际未使用
        """
        分子-文本多模态大语言模型
        
        参数:
            qwen_model_name: Qwen基础模型路径
            d_mol: 分子特征维度（兼容性参数）
        """
        # 加载Qwen模型的配置文件
        config = PretrainedConfig.from_pretrained(qwen_model_name)
        super().__init__(config)

        # ---- 1. 加载预训练的Qwen LLM ----
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
        
        # 添加分子特殊标记
        self.extra_tokens = ["<mol_start>", "<mol_end>"]
        self.tokenizer.add_tokens(self.extra_tokens)

        # 加载基础语言模型
        self.model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            torch_dtype=torch.float32,  # 使用半精度浮点数
            device_map="auto"           # 自动设备映射
        )
        
        # 调整词表大小以包含新添加的特殊标记
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # 获取特殊标记的ID
        self.start_id = self.tokenizer.convert_tokens_to_ids("<mol_start>")
        self.end_id = self.tokenizer.convert_tokens_to_ids("<mol_end>")

        # 获取LLM的嵌入维度
        self.d_llm = self.model.get_input_embeddings().weight.shape[1]

        # ---- 2. 分子编码器和投影器 ----
        # 加载预训练的分子编码器（SMI-TED）
        self.mol_encoder = load_smi_ted(
            folder='/zengdaojian/zhangjia/BioLatent/smi-ted/smi-ted/inference/smi_ted_light',
            ckpt_filename='/zengdaojian/zhangjia/BioLatent/smi-ted/smi-ted-Light_40.pt'
        )
        
        # 初始化投影器
        self.projector = QueryAttentionProjector()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        **kwargs,
    ):
        """
        前向传播（训练和推理）
        
        注意: 必须通过kwargs传入smiles参数
        """
        # ============================
        # 1. 提取SMILES字符串
        # ============================
        smiles_list = kwargs.pop("smiles", None)
        if smiles_list is None:
            raise ValueError("训练/推理时必须提供smiles参数")

        B = len(smiles_list)  # 批次大小
        device = self.model.device  # 设备（CPU/GPU）

        # ============================
        # 2. 分子编码（编码器被冻结）
        # ============================
        with torch.no_grad():  # 冻结分子编码器，不计算梯度
            # 编码SMILES字符串为分子特征
            mol_emb = self.mol_encoder.encode(
                smiles_list, return_torch=True
            ).to(device)  # [B, L_mol, d_mol]

        # 将分子特征投影到LLM嵌入空间
        mol_emb_llm = self.projector(mol_emb)  # [B, num_queries, d_llm]

        # 获取LLM的嵌入层
        embed = self.model.get_input_embeddings()
        
        # 创建分子开始标记的嵌入
        start_emb = embed(
            torch.tensor([self.start_id], device=device)
        ).expand(B, 1, -1)  # [B, 1, d_llm]
        
        # 创建分子结束标记的嵌入
        end_emb = embed(
            torch.tensor([self.end_id], device=device)
        ).expand(B, 1, -1)  # [B, 1, d_llm]

        # 拼接分子前缀：<mol_start> + 分子特征 + <mol_end>
        mol_prefix = torch.cat([start_emb, mol_emb_llm, end_emb], dim=1)
        L_mol = mol_prefix.size(1)  # 分子前缀的长度

        # ============================
        # 3. 文本嵌入
        # ============================
        if inputs_embeds is None:
            # 如果未提供嵌入，通过token IDs计算
            text_emb = embed(input_ids)  # [B, L_text, d_llm]
        else:
            # 使用提供的嵌入
            text_emb = inputs_embeds

        # ============================
        # 4. 融合分子和文本嵌入
        # ============================
        fused_embeds = torch.cat([mol_prefix, text_emb], dim=1).to(self.model.dtype)

        # ============================
        # 5. 注意力掩码处理
        # ============================
        if attention_mask is not None:
            # 为分子部分创建全1的掩码
            mol_mask = torch.ones(B, L_mol, device=device, dtype=attention_mask.dtype)
            # 拼接分子掩码和文本掩码
            fused_attention_mask = torch.cat([mol_mask, attention_mask], dim=1)
        else:
            fused_attention_mask = None

        # ============================
        # 6. 位置ID处理
        # ============================
        if position_ids is not None:
            # 为分子部分创建连续位置ID
            mol_pos = torch.arange(L_mol, device=device).unsqueeze(0).expand(B, -1)
            # 文本部分的位置ID从L_mol开始
            fused_position_ids = torch.cat([mol_pos, position_ids + L_mol], dim=1)
        else:
            fused_position_ids = None

        # ============================
        # 7. 调用Qwen基础模型
        # ============================
        outputs = self.model(
            inputs_embeds=fused_embeds,           # 融合后的嵌入
            attention_mask=fused_attention_mask,  # 融合后的注意力掩码
            position_ids=fused_position_ids,      # 融合后的位置ID
            past_key_values=past_key_values,      # 用于加速生成的缓存
            use_cache=use_cache,                  # 是否使用缓存
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=True,                     # 返回字典格式
        )

        # 获取模型输出logits
        logits = outputs.logits  # [B, T, vocab_size]

        # ============================
        # 8. 计算损失（训练时使用）
        # ============================
        loss = None
        if labels is not None:
            # 调试信息：检查形状
            print(f"logits形状: {logits.shape}")
            print(f"labels形状: {labels.shape}")
            
            # 标准因果语言建模的shift方式
            shift_logits = logits[:, :-1, :].contiguous()  # 去掉最后一个token
            shift_labels = labels[:, 1:].contiguous()      # 去掉第一个token
            
            # 计算交叉熵损失
            # 注意：分子部分对应的标签应为-100（在数据处理时已设置）
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),  # 展平: [B*(T-1), vocab]
                shift_labels.view(-1),                         # 展平: [B*(T-1)]
                ignore_index=-100,                             # 忽略标签为-100的位置
            )

        # ============================
        # 9. 返回输出
        # ============================
        return CausalLMOutputWithPast(
            loss=loss,                        # 损失值（训练时）
            logits=logits,                    # 预测logits
            past_key_values=outputs.past_key_values,  # 缓存（用于生成）
            hidden_states=outputs.hidden_states,      # 隐藏状态
            attentions=outputs.attentions,            # 注意力权重
        )

# 在 model_new.py 的 Qwen3MoleculeLLM 类中添加


    # def generate(
    #     self,
    #     smiles_list: list[str],
    #     input_ids: torch.Tensor,  # tokenized prompt (B, prompt_len)
    #     max_new_tokens: int = 200,
    #     temperature: float = 0.7,
    #     top_p: float = 0.9,
    #     do_sample: bool = True,
    #     **kwargs
    # ) -> torch.Tensor:
    #     self.eval()
    #     batch_size = input_ids.size(0)
    #     device = input_ids.device

    #     # Project SMILES to queries (assume projector returns (B, num_queries, dim))
    #     queries = self.projector(smiles_list).to(device)  # 需确认projector支持list

    #     # 初始化 generated_ids = input_ids
    #     generated_ids = input_ids.clone()

    #     # 假设 LLM 是 CausalLM，需逐步生成
    #     # 注意：queries作为prefix，类似past_key_values，但需自定义loop因为queries是fixed
    #     for _ in range(max_new_tokens):
    #         # Forward: 获取当前logits (queries + generated_ids)
    #         with torch.no_grad():
    #             logits = self.forward(smiles_list, generated_ids)  # 复用原有forward

    #         # 取最后一个token的logits (B, 1, vocab)
    #         next_logits = logits[:, -1, :]

    #         # Sampling
    #         if do_sample:
    #             next_logits = next_logits / temperature
    #             probs = torch.softmax(next_logits, dim=-1)
    #             # top_p nucleus sampling
    #             sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    #             cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    #             mask = cumulative_probs > top_p
    #             mask[..., 0] = False  # 至少保持top1
    #             sorted_probs[mask] = 0
    #             probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    #             next_token = torch.gather(sorted_indices, -1, torch.multinomial(probs, num_samples=1))
    #         else:
    #             next_token = next_logits.argmax(dim=-1, keepdim=True)

    #         # Append
    #         generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    #         # Early stop if all EOS
    #         if torch.all(next_token == self.tokenizer.eos_token_id):
    #             break

    #     return generated_ids



    @torch.no_grad()
    def generate(
        self,
        smiles_list: List[str],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ):
        """
        稳定版 generate（支持 molecule prefix + Qwen）
        """

        device = input_ids.device
        B = input_ids.size(0)

        # =========================================================
        # 0. tokenizer 安全兜底（只需一次，但多做一次不出错）
        # =========================================================
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # =========================================================
        # 1. 文本 embedding
        # =========================================================
        text_embeds = self.model.get_input_embeddings()(input_ids)
        # shape: [B, L_text, hidden]

        L_text = text_embeds.size(1)

        # =========================================================
        # 2. 分子 query prefix（必须是 Tensor！）
        # =========================================================
        # projector 输入必须是 tensor / batch tensor
        # 这里假设 projector 已经内部处理 SMILES → embedding
        with torch.no_grad():  # 冻结分子编码器，不计算梯度
    # 编码SMILES字符串为分子特征
            mol_emb = self.mol_encoder.encode(
                smiles_list, return_torch=True
            ).to(device)  # [B, L_mol, d_mol]
        mol_embeds = self.projector(mol_emb)
        mol_embeds = mol_embeds.to(device)

        # 确保 batch 对齐
        assert mol_embeds.size(0) == B, "Batch size mismatch"

        L_mol = mol_embeds.size(1)

        # =========================================================
        # 3. 拼接 inputs_embeds
        # =========================================================
        inputs_embeds = torch.cat([mol_embeds, text_embeds], dim=1)
        # shape: [B, L_mol + L_text, hidden]

        total_len = inputs_embeds.size(1)

        # =========================================================
        # 4. attention_mask（必须和 inputs_embeds 对齐）
        # =========================================================
        if attention_mask is None:
            text_mask = torch.ones(B, L_text, device=device)
        else:
            text_mask = attention_mask.to(device)

        mol_mask = torch.ones(B, L_mol, device=device)
        fused_attention_mask = torch.cat([mol_mask, text_mask], dim=1).long()

        # =========================================================
        # 5. position_ids（❗关键❗inputs_embeds 必须显式给）
        # =========================================================
        position_ids = torch.arange(
            total_len,
            device=device
        ).unsqueeze(0).expand(B, -1)

        # =========================================================
        # 6. 最稳妥的 generate 调用
        # =========================================================
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=fused_attention_mask,
            position_ids=position_ids,              # ⭐ 必须
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )

        return outputs



# ============================
# 3. 使用示例
# ============================
if __name__ == "__main__":
    # 初始化模型
    model = Qwen3MoleculeLLM(
        qwen_model_name="/zengdaojian/zhangjia/BioLatent/Qwen4B",
    ).cuda()
    
    tokenizer = model.tokenizer

    # 示例文本
    texts = [
        "Please describe the functional groups of this molecule.",
        "Please describe the functional groups of this molecule  123  yes"
    ]
    
    # 文本编码
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].cuda()
    attention_mask = enc["attention_mask"].cuda()

    # 示例SMILES
    smiles_list = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # 阿司匹林
        "CC(=O)OC1=CC=CC=C1C(=O)O"
    ]
    
    # 示例标签（实际训练时会来自数据集）
    Labels = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "CC(=O)OC1=CC=CC=C1C(=O)O"
    ]

    # 前向传播
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        smiles=smiles_list
    )
    
    # 输出logits形状
    print("模型输出logits形状:", outputs.logits.shape)