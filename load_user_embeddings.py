import torch
import torch.nn.functional as F

def load_user_embeddings(pt_path):
    """
    從 .pt 檔載入 user embedding tensor
    """
    state_dict = torch.load(pt_path, map_location='cpu')
    if 'user_embedding.weight' in state_dict:
        return state_dict['user_embedding.weight']
    else:
        raise KeyError("'user_embedding.weight' not found in checkpoint.")

def get_hard_users(pt_path, user_with_top_item, user_without_top_item, top_percent=0.1):
    """
    計算 hard users（直接傳入已分好的使用者 index）
    
    Args:
        pt_path (str): 模型檔案路徑 (.pt)
        user_with_top_item (list[int] 或 set[int]): 有買熱門商品的使用者 index
        user_without_top_item (list[int] 或 set[int]): 沒買熱門商品的使用者 index
        top_percent (float): 取前多少比例的 hard user（預設 0.1 = 10%）

    Returns:
        list[int]: 被選中的 hard user index
    """
    # 載入 embedding
    user_embeddings = load_user_embeddings(pt_path)

    # 轉成 list，確保索引一致
    with_list = list(user_with_top_item)
    without_list = list(user_without_top_item)

    with_embeds = user_embeddings[with_list]
    without_embeds = user_embeddings[without_list]

    hard_scores = []
    for i, u_without in enumerate(without_embeds):
        u_without = u_without.unsqueeze(0)  # [1, D]
        cos_sim = F.cosine_similarity(u_without, with_embeds)  # [N_with]
        cos_dist = 1 - cos_sim
        max_dist = cos_dist.max().item()
        hard_scores.append((without_list[i], max_dist))

    # 排序 & 取前 top_percent
    hard_scores.sort(key=lambda x: x[1], reverse=True)
    cutoff = int(len(hard_scores) * top_percent)
    hard_users = [uid for uid, _ in hard_scores[:cutoff]]
    top_hard_scores = hard_scores[:cutoff]
    
    print("Top {}% Hard Users and their max cosine distances:".format(int(top_percent*100)))
    for uid, dist in top_hard_scores:
        print(f"User {uid}: cos distance = {dist:.4f}")

    return hard_users
