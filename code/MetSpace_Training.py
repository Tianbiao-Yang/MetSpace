import os

# 设置环境变量以禁用tokenizers的并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedTokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
import numpy as np

# 定义SMILES数据集类
class SMILESDataset(Dataset):
    def __init__(self, tokenizer, smiles_list):
        self.tokenizer = tokenizer
        self.smiles_list = smiles_list

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smile = self.smiles_list[idx]
        encoded_smile = self.tokenizer(smile, add_special_tokens=True, return_tensors="pt")
        return encoded_smile.input_ids[0], encoded_smile.attention_mask[0]

# 加载SMILES数据
def load_smiles(file_path):
    with open(file_path, 'r') as file:
        return file.read().splitlines()

# 创建字符级分词器并训练
def train_tokenizer(smiles_data, vocab_size=1000):
    tokenizer = ByteLevelBPETokenizer()
    special_tokens = ["<pad>", "<eos>", "<bos>", "<unk>"]
    tokenizer.train(files=[smiles_data], vocab_size=vocab_size, min_frequency=2, special_tokens=special_tokens)
    tokenizer.save("smiles_tokenizer.json")
    return PreTrainedTokenizerFast(tokenizer_file="smiles_tokenizer.json",
                                    eos_token="<eos>",
                                    bos_token="<bos>",
                                    unk_token="<unk>",
                                    pad_token="<pad>",
                                    mask_token=None)

# 创建数据加载器
def create_dataloader(smiles_data, tokenizer, batch_size=128):
    smiles_dataset = SMILESDataset(tokenizer, smiles_data)

    def collate_fn(batch):
        input_ids, attention_mask = zip(*batch)
        pad_token_id = tokenizer.pad_token_id
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        return input_ids, attention_mask

    return DataLoader(smiles_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=10)

# 训练模型
def train_model(model, dataloader, optimizer, device, epochs, save_dir):
    model.train()
    for epoch in range(epochs):
        all_loss = []
        for batch in dataloader:
            input_ids, attention_mask = [b.to(device) for b in batch]
            outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            all_loss.append(loss.item())
        
        loss_mean = np.mean(all_loss)
        print(f"Epoch: {epoch}, Average Loss: {loss_mean}")

        # 保存模型和分词器
        model.save_pretrained(os.path.join(save_dir, f'epoch_{epoch}'))
        tokenizer.save_pretrained(os.path.join(save_dir, f'epoch_{epoch}'))


def calculate_score(model, tokenizer, smile):
    inputs = tokenizer(smile, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss.item()
        return -loss  # 使用负的损失值作为评分，越大表示生成质量越好

# # 过滤生成的SMILES以优化质量
# def filter_smiles(generated_smiles, model, tokenizer, quality_threshold=0.5):
#     filtered_smiles = []
#     for smile in generated_smiles:
#         score = calculate_score(model, tokenizer, smile)
#         if score >= quality_threshold:
#             filtered_smiles.append((smile, score))  # 保存SMILES和其评分
#     return filtered_smiles

# 主函数
if __name__ == "__main__":
    # 文件路径
    chembl_file = '../data/HMDB_Database.txt'
    hmdb_file = '../data/BAs_set.txt'

    # 加载SMILES数据
    chembl_data = load_smiles(chembl_file)
    hmdb_data = load_smiles(hmdb_file)

    # 创建分词器
    tokenizer = train_tokenizer(chembl_file, vocab_size=1000)

    # 创建数据加载器
    chembl_loader = create_dataloader(chembl_data, tokenizer)
    hmdb_loader = create_dataloader(hmdb_data, tokenizer)

    # 初始化模型
    config = GPT2Config(vocab_size=tokenizer.vocab_size, n_layer=6, n_embd=256, n_head=4)
    model = GPT2LMHeadModel(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 准备优化器
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.000001)

    # 在HMDB数据集上预训练
    print("Pretraining on HMDB database...")
    train_model(model, chembl_loader, optimizer, device, epochs=400, save_dir='../saved_models/gpt_pretrained')

    # 在BAs数据集上微调
    print("Finetuning on BAs set...")
    train_model(model, hmdb_loader, optimizer, device, epochs=400, save_dir='../saved_models/gpt_finetuned')

#     # 生成新的SMILES
#     seed_smile = "C[C@H]1CC[C@H]2[C@@H]3CC[C@@H]4CCCC[C@]4(C)[C@H]3CC[C@]12C"  # 使用一个简单的种子 SMILES
#     num_samples = 10  # 设定生成的SMILES数量
#     generated_smiles = generate_multiple_smiles(model, tokenizer, seed_smile, num_samples)

#     # 过滤生成的SMILES以优化质量
#     quality_threshold = -0.5  # 根据需要设置质量阈值
#     filtered_smiles_with_scores = filter_smiles(generated_smiles, model, tokenizer, quality_threshold)

#     # 保存生成的SMILES及其评分到文件
#     with open('generated_smiles.txt', 'w') as f:
#         for smile, score in filtered_smiles_with_scores:
#             f.write(f"{smile}, {score}\n")

#     print(f"Generated {len(filtered_smiles_with_scores)} unique SMILES with scores and saved to 'generated_smiles.txt'.")
