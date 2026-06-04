"""
peoples_daily 数据集处理模块
支持 BIO 标签体系，处理子词对齐问题
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
ENTITY_TYPES = ["PER", "ORG", "LOC"]

def build_label_schema() -> tuple[list, dict, dict]:
    """构建 BIO 标签体系。"""
    labels = ["O"]
    for entity_type in ENTITY_TYPES:
        labels.append(f"B-{entity_type}")
        labels.append(f"I-{entity_type}")
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return labels, label2id, id2label

def load_records(split: str, data_dir: Optional[Path] = None) -> list:
    """加载数据集记录。"""
    d = data_dir if data_dir else DATA_DIR
    with open(d / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)

class PeoplesDailyDataset(Dataset):
    """peoples_daily 的 PyTorch Dataset。"""
    
    def __init__(self, records: list, tokenizer: BertTokenizer, label2id: dict, max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        text = record["text"]
        entities = record.get("entities", [])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )
        
        offset_mapping = encoding["offset_mapping"]
        aligned_labels = [-100] * self.max_length
        
        for entity in entities:
            start = entity["start_idx"]
            end = entity["end_idx"]
            entity_type = entity["type"]
            
            for i in range(self.max_length):
                token_start, token_end = offset_mapping[i]
                if token_start == token_end == 0:
                    continue
                
                if token_start < end and token_end > start:
                    if token_start == start:
                        label = f"B-{entity_type}"
                    else:
                        label = f"I-{entity_type}"
                    aligned_labels[i] = self.label2id.get(label, self.label2id["O"])
        
        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)
        
        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(encoding["token_type_ids"], dtype=torch.long),
            "labels": labels_tensor,
        }

def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int,
    max_length: int,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试 DataLoader。"""
    train_records = load_records("train", data_dir)
    val_records = load_records("validation", data_dir)
    test_records = load_records("test", data_dir)
    
    print(f"数据集规模：训练={len(train_records)}，验证={len(val_records)}，测试={len(test_records)}")
    
    train_ds = PeoplesDailyDataset(train_records, tokenizer, label2id, max_length)
    val_ds = PeoplesDailyDataset(val_records, tokenizer, label2id, max_length)
    test_ds = PeoplesDailyDataset(test_records, tokenizer, label2id, max_length)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader
