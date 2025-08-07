from pathlib import Path

# 重新寫入更新後的 Python 程式碼（包含 hard user 計算邏輯）
updated_code = """import torch
import torch.nn.functional as F
import json

def load_user_embeddings(pt_path):
    '''
    讀取儲存於 PyTorch .pt 檔案中的使用者嵌入向量 (user embeddings)
    '''
    state_dict = torch.load(pt_path, map_location='cpu')
    if 'user_embedding.weight' in state_dict:
        user_embeddings = state_dict['user_embedding.weight']
        print(f"✅ Loaded user embeddings. Shape: {user_embeddings.shape}")
        return user_embeddings
    else:
        raise KeyError("❌ 'user_embedding.weight' not found in checkpoint.")

def load_user_sets(json_path):
    '''
    載入 user_with_top_item 和 user_without_top_item 的 ID (index) 清單
    '''
    with open(json_path, 'r') as f:
        data = json.load(f)
    return set(data['user_with_top_item']), set(data['user_without_top_item'])

def compute_hard_users(user_embeddings, user_with_top_item, user_without_top_item, top_percent=0.1):
    '''
    計算每位 user_without_top_item 使用者與所有 user_with_top_item 使用者的最大 cosine distance，
    然後選出 max 10% 距離最大的 user → 為 hard users。
    '''
    with_embeds = user_embeddings[list(user_with_top_item)]
    without_embeds = user_embeddings[list(user_without_top_item)]

    hard_scores = []
    for i, u_without in enumerate(without_embeds):
        u_without = u_without.unsqueeze(0)  # [1, D]
        cos_sim = F.cosine_similarity(u_without, with_embeds)  # [N_with]
        cos_dist = 1 - cos_sim
        max_dist = cos_dist.max().item()
        hard_scores.append((list(user_without_top_item)[i], max_dist))

    # 排序取前 top_percent
    hard_scores.sort(key=lambda x: x[1], reverse=True)
    cutoff = int(len(hard_scores) * top_percent)
    hard_users = [uid for uid, _ in hard_scores[:cutoff]]
    return hard_users

if __name__ == "__main__":
    pt_path = "CD_Kitchen.pt"  # 模型檔案
    json_path = "user_sets.json"  # 包含 user_with_top_item 和 user_without_top_item 的 index

    embeddings = load_user_embeddings(pt_path)
    user_with_top_item, user_without_top_item = load_user_sets(json_path)

    hard_users = compute_hard_users(embeddings, user_with_top_item, user_without_top_item)
    print(f"Hard users (top 10%): {hard_users}")
"""

# 儲存成可執行的 Python 檔案
py_file_path = "/mnt/data/load_user_embeddings.py"
Path(py_file_path).write_text(updated_code)

py_file_path  # 傳回下載路徑給使用者
