import os
import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import random

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型和分词器
model_dir = '../saved_models/gpt_finetuned/epoch_399'  # 修改为你保存模型的路径
# model_dir = '../saved_models/gpt_pretrained/epoch_399'  # 修改为你保存模型的路径
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir, eos_token="<eos>", bos_token="<bos>", unk_token="<unk>", pad_token="<pad>")
model = GPT2LMHeadModel.from_pretrained(model_dir)
model = model.to(device)

# 计算SMILES的评分
def calculate_score(model, tokenizer, smile):
    inputs = tokenizer(smile, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=-1)

    score_sum = 0
    valid_token_count = 0
    for i in range(inputs.size(1) - 1):
        true_token_id = inputs[0, i + 1].item()
        predicted_probability = probabilities[0, i, true_token_id].item()
        score_sum += torch.log(torch.tensor(predicted_probability + 1e-8)).item()
        valid_token_count += 1

    average_score = torch.exp(torch.tensor(score_sum / valid_token_count)).item()
    normalized_score = max(0, min(1, average_score))

    return normalized_score
    
# 从文件中加载SMILES
def load_smiles_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().splitlines()

# 主函数
if __name__ == "__main__":
    # 文件路径
    smiles_file = '../data/Smiles_Input.txt'  # 修改为包含SMILES的文件路径
    output_file = '../result/Smiles_Input_scores_test.txt'  # 输出文件路径
    # output_file = '../result/BILELIB19_LIPIDMAPS_pretrained_scores.txt'  # 输出文件路径

    # 加载SMILES数据
    smiles_list = load_smiles_from_file(smiles_file)
    
    # smiles_list = ['CC(C)C[C@H]1CC[C@H]2[C@@H]3CC[C@@H]4CCCC[C@]4(C)[C@H]3CC[C@]12C']
    
    results = []  # 用于存储结果的列表

    # 计算并输出每个SMILES的评分
    for smile in smiles_list:
        try:
            score = calculate_score(model, tokenizer, smile)
            results.append((smile, score))  # 将结果添加到列表中
            print(f"SMILES: {smile}, Score: {score}")
        except Exception as e:
            print(f"Error processing {smile}: {e}")  # 输出错误信息

    # 将结果写入文件
    with open(output_file, 'w') as f:
        for smile, score in results:
            f.write(f"{smile}\t{score}\n")  # 将SMILES和评分写入文件，使用制表符分隔

    print(f"Results saved to {output_file}")
