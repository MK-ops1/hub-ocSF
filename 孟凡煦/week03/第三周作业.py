import torch
import torch.nn as nn
import torch.optim as optim
import random

vocab = {
    "[pad]": 0, 
    "[unk]": 1, 
    "你": 2, 
    "好": 3, 
    "中": 4, 
    "国": 5,
    "学": 6, 
    "习": 7, 
    "啊": 8 
}
UNK_ID = vocab["[unk]"]
# 文字转ID
def text2ids(text):
    return [vocab.get(c, UNK_ID) for c in text]

def generate_data(num_samples=1000):
    texts, labels = [], []
    chars = [c for c in vocab.keys() if c not in ["[pad]", "[unk]"]]  
    for _ in range(num_samples):
        pos = random.randint(0, 4)  # 你在0-4位
        text = []
        for i in range(5):
            if i == pos:
                text.append("你")
            else:
                # 随机选其他字
                text.append(random.choice([c for c in chars if c != "你"]))
        texts.append("".join(text))
        labels.append(pos)
    return texts, labels

# 生成1000条训练数据
texts, labels = generate_data(num_samples=1000)
# 转换为模型输入
train_x = [text2ids(t) for t in texts]
X = torch.LongTensor(train_x)
Y = torch.LongTensor(labels)

SELECT_MODEL = "rnn"  # 切换为 "lstm" 即可

class MyModel(nn.Module):
    def __init__(self, model_type="rnn"):
        super().__init__()
        self.model_type = model_type
        self.embed = nn.Embedding(len(vocab), 8, padding_idx=0)
        
        if model_type == "rnn":
            self.rnn = nn.RNN(input_size=8, hidden_size=16, batch_first=True)
        elif model_type == "lstm":
            self.lstm = nn.LSTM(input_size=8, hidden_size=16, batch_first=True)
        
        self.fc = nn.Linear(16, 5)

    def forward(self, x):
        x = self.embed(x)
        if self.model_type == "rnn":
            _, h = self.rnn(x)
        elif self.model_type == "lstm":
            _, (h, c) = self.lstm(x)
        out = self.fc(h.squeeze(0))
        return out

# 初始化模型
model = MyModel(model_type=SELECT_MODEL)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# ===================== 训练模型 =====================
print(f"===== 训练模型：{SELECT_MODEL.upper()} | 样本数：1000 =====")
for epoch in range(50):
    optimizer.zero_grad()
    predict = model(X)
    loss = loss_fn(predict, Y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1} | 损失: {loss.item():.4f}")

# ===================== 测试 =====================
test_text = "好你中国啊"  
test_input = torch.LongTensor([text2ids(test_text)])

model.eval()
with torch.no_grad():
    result = model(test_input)
    prediction = result.argmax(dim=1).item()

print(f"\n测试句子: {test_text}")
print(f"模型预测 '你' 的位置是: {prediction}")
