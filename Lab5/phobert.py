import torch
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm  # Thanh tiến trình

phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

DIR = "D:/vncorenlp/extracting_segmented.json"
text_tensor = []
label_tensor = []

with open(DIR, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Đếm tổng số dòng cần xử lý
total_sentences = sum(len(value) for value in data.values())

# Duyệt dữ liệu kèm thanh tiến trình
with torch.no_grad():
    with tqdm(total=total_sentences, desc="Processing", unit="sample") as pbar:
        for key, value in data.items():
            for index in range(len(value)):
                # Xử lí text
                input_ids = torch.tensor([tokenizer.encode(data[key][index]['text'][0])])
                outputs = phobert(input_ids)
                last_hidden_state = outputs[0]
                cls_embedding = last_hidden_state[:, 0, :]  # (1, 768)
                text_tensor.append(cls_embedding)

                # Xử lí label
                label_tensor.append(data[key][index]['label'])

                pbar.update(1)

X = torch.stack(text_tensor, dim=0)  # (n, 768)
y = torch.tensor(label_tensor)       # (n, 1)

# Lưu X và y
torch.save(X, 'D:/vncorenlp/X.pt')
torch.save(y, 'D:/vncorenlp/y.pt')

print("✅ Xử lý và lưu hoàn tất!")
