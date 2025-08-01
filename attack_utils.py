import torch, math
import random
import logging
import pandas as pd
from model import Model 
from math import ceil
def attack_all_items(df):
    df = df.copy()
    item_counts = df["item"].value_counts()
    total_items = len(item_counts)

    sorted_items = item_counts.index.tolist()
    sorted_counts = item_counts.values.tolist()

    total_interactions = sum(sorted_counts)
    target_per_group = total_interactions / 10

    groups = []
    current_group = []
    current_sum = 0

    for item, count in zip(sorted_items, sorted_counts):
        current_group.append(item)
        current_sum += count
        if current_sum >= target_per_group:
            groups.append(current_group)
            current_group = []
            current_sum = 0

    if current_group:
        groups.append(current_group)

    while len(groups) < 10:
        groups.append([])

    multipliers = [0.5 - 0.05 * i for i in range(10)]

    all_users = set(df["user"].unique())
    new_rows = []

    for group_items, ratio in zip(groups, multipliers):
        for item in group_items:
            n = item_counts[item]
            num_to_inject = int(n * ratio)
            current_users = set(df[df["item"] == item]["user"])
            candidate_users = list(all_users - current_users)

            if not candidate_users or num_to_inject == 0:
                continue

            inject_users = random.sample(candidate_users, min(num_to_inject, len(candidate_users)))
            for i, user in enumerate(inject_users):
                user_interactions = df[df["user"] == user]
                if len(user_interactions) < 2:
                    continue
                second_last_time = user_interactions.iloc[-2]["timestamp"]
                inject_timestamp = second_last_time - 1e-4

                new_rows.append({
                    "user": user,
                    "item": item,
                    "timestamp": inject_timestamp,
                    "click": 1.0,
                    "is_target": 1
                })

    attacked_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    attacked_df.sort_values(by="timestamp", inplace=True)
    attacked_df.reset_index(drop=True, inplace=True)

    print(f"🔥 Max interaction count for single item: {item_counts.max()}")
    print(f"✅ Total items: {total_items}")
    print(f"🧪 Injected {len(new_rows)} fake interactions.")
    for i, g in enumerate(groups):
        total_group_count = sum(item_counts[item] for item in g)
        print(f"📦 Group {i+1}: {len(g)} items, total count = {total_group_count}")

    return attacked_df

def attack_top20_percent_items(df):
    df = df.copy()
    item_counts = df["item"].value_counts()
    total_items = len(item_counts)

    sorted_items = item_counts.index.tolist()
    sorted_counts = item_counts.values.tolist()

    top_20_len = max(1, int(total_items * 0.2))
    top_items = sorted_items[:top_20_len]
    top_counts = [item_counts[item] for item in top_items]

    total_interactions = sum(top_counts)
    target_per_group = total_interactions / 10

    groups = []
    current_group = []
    current_sum = 0

    for item, count in zip(top_items, top_counts):
        current_group.append(item)
        current_sum += count
        if current_sum >= target_per_group:
            groups.append(current_group)
            current_group = []
            current_sum = 0

    if current_group:
        groups.append(current_group)

    while len(groups) < 10:
        groups.append([])

    multipliers = [0.5 - 0.05 * i for i in range(10)]

    all_users = set(df["user"].unique())
    new_rows = []

    for group_items, ratio in zip(groups, multipliers):
        for item in group_items:
            n = item_counts[item]
            num_to_inject = int(n * ratio)
            current_users = set(df[df["item"] == item]["user"])
            candidate_users = list(all_users - current_users)

            if not candidate_users or num_to_inject == 0:
                continue

            inject_users = random.sample(candidate_users, min(num_to_inject, len(candidate_users)))
            for i, user in enumerate(inject_users):
                user_interactions = df[df["user"] == user]
                if len(user_interactions) < 2:
                    continue
                second_last_time = user_interactions.iloc[-2]["timestamp"]
                inject_timestamp = second_last_time - 1e-4

                new_rows.append({
                    "user": user,
                    "item": item,
                    "timestamp": inject_timestamp,
                    "click": 1.0,
                    "is_target": 1
                })

    attacked_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    attacked_df.sort_values(by="timestamp", inplace=True)
    attacked_df.reset_index(drop=True, inplace=True)

    print(f"🔥 Max interaction count for single item: {item_counts.max()}")
    print(f"✅ Top 20% items: {top_20_len}")
    print(f"🧪 Injected {len(new_rows)} fake interactions.")
    for i, g in enumerate(groups):
        total_group_count = sum(item_counts[item] for item in g)
        print(f"📦 Group {i+1}: {len(g)} items, total count = {total_group_count}")

    return attacked_df


def attack_source_all_items(df):
    df = df.copy()
    df = df[df["is_target"] == 0]  # 專攻 source domain

    item_counts = df["item"].value_counts()
    total_items = len(item_counts)

    sorted_items = item_counts.index.tolist()
    sorted_counts = item_counts.values.tolist()

    total_interactions = sum(sorted_counts)
    target_per_group = total_interactions / 10

    groups = []
    current_group = []
    current_sum = 0

    for item, count in zip(sorted_items, sorted_counts):
        current_group.append(item)
        current_sum += count
        if current_sum >= target_per_group:
            groups.append(current_group)
            current_group = []
            current_sum = 0

    if current_group:
        groups.append(current_group)
    while len(groups) < 10:
        groups.append([])

    multipliers = [0.5 - 0.05 * i for i in range(10)]

    all_users = set(df["user"].unique())
    new_rows = []

    for group_items, ratio in zip(groups, multipliers):
        for item in group_items:
            n = item_counts[item]
            num_to_inject = int(n * ratio)
            current_users = set(df[df["item"] == item]["user"])
            candidate_users = list(all_users - current_users)

            if not candidate_users or num_to_inject == 0:
                continue

            inject_users = random.sample(candidate_users, min(num_to_inject, len(candidate_users)))
            for user in inject_users:
                user_interactions = df[df["user"] == user]
                if len(user_interactions) < 2:
                    continue
                second_last_time = user_interactions.iloc[-2]["timestamp"]
                inject_timestamp = second_last_time - 1e-4

                new_rows.append({
                    "user": user,
                    "item": item,
                    "timestamp": inject_timestamp,
                    "click": 1.0,
                    "is_target": 0  # Source domain
                })

    attacked_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    attacked_df.sort_values(by="timestamp", inplace=True)
    attacked_df.reset_index(drop=True, inplace=True)

    print(f"[SOURCE ATTACK] 🔥 Max interaction count for single item: {item_counts.max()}")
    print(f"[SOURCE ATTACK] ✅ Total items: {total_items}")
    print(f"[SOURCE ATTACK] 🧪 Injected {len(new_rows)} fake interactions.")
    for i, g in enumerate(groups):
        total_group_count = sum(item_counts[item] for item in g)
        print(f"[SOURCE ATTACK] 📦 Group {i+1}: {len(g)} items, total count = {total_group_count}")

    return attacked_df


def attack_source_top20_percent_items(df):
    df = df.copy()
    df = df[df["is_target"] == 0]  # 只攻擊 source domain

    item_counts = df["item"].value_counts()
    total_items = len(item_counts)

    sorted_items = item_counts.index.tolist()
    sorted_counts = item_counts.values.tolist()

    top_20_len = max(1, int(total_items * 0.2))
    top_items = sorted_items[:top_20_len]
    top_counts = [item_counts[item] for item in top_items]

    total_interactions = sum(top_counts)
    target_per_group = total_interactions / 10

    groups = []
    current_group = []
    current_sum = 0

    for item, count in zip(top_items, top_counts):
        current_group.append(item)
        current_sum += count
        if current_sum >= target_per_group:
            groups.append(current_group)
            current_group = []
            current_sum = 0

    if current_group:
        groups.append(current_group)
    while len(groups) < 10:
        groups.append([])

    multipliers = [0.5 - 0.05 * i for i in range(10)]

    all_users = set(df["user"].unique())
    new_rows = []

    for group_items, ratio in zip(groups, multipliers):
        for item in group_items:
            n = item_counts[item]
            num_to_inject = int(n * ratio)
            current_users = set(df[df["item"] == item]["user"])
            candidate_users = list(all_users - current_users)

            if not candidate_users or num_to_inject == 0:
                continue

            inject_users = random.sample(candidate_users, min(num_to_inject, len(candidate_users)))
            for user in inject_users:
                user_interactions = df[df["user"] == user]
                if len(user_interactions) < 2:
                    continue
                second_last_time = user_interactions.iloc[-2]["timestamp"]
                inject_timestamp = second_last_time - 1e-4

                new_rows.append({
                    "user": user,
                    "item": item,
                    "timestamp": inject_timestamp,
                    "click": 1.0,
                    "is_target": 0  # Source domain
                })

    attacked_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    attacked_df.sort_values(by="timestamp", inplace=True)
    attacked_df.reset_index(drop=True, inplace=True)

    print(f"[SOURCE ATTACK] 🔥 Max interaction count for single item: {item_counts.max()}")
    print(f"[SOURCE ATTACK] ✅ Top 20% items: {top_20_len}")
    print(f"[SOURCE ATTACK] 🧪 Injected {len(new_rows)} fake interactions.")
    for i, g in enumerate(groups):
        total_group_count = sum(item_counts[item] for item in g)
        print(f"[SOURCE ATTACK] 📦 Group {i+1}: {len(g)} items, total count = {total_group_count}")

    return attacked_df



def attack_source_cold_start_items(df):
    df = df.copy()
    df = df[df["is_target"] == 0]  # 攻擊 source domain

    item_counts = df["item"].value_counts()
    cold_items = item_counts[item_counts == 1].index.tolist()

    all_users = set(df["user"].unique())
    new_rows = []

    for item in cold_items:
        original_row = df[df["item"] == item].iloc[0]
        current_users = set([original_row["user"]])
        candidate_users = list(all_users - current_users)

        if not candidate_users:
            continue

        inject_user = random.choice(candidate_users)
        user_interactions = df[df["user"] == inject_user]
        if len(user_interactions) < 2:
            continue

        second_last_time = user_interactions.iloc[-2]["timestamp"]
        inject_timestamp = second_last_time - 1e-4

        new_rows.append({
            "user": inject_user,
            "item": item,
            "timestamp": inject_timestamp,
            "click": 1.0,
            "is_target": 0
        })

    attacked_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    attacked_df.sort_values(by="timestamp", inplace=True)
    attacked_df.reset_index(drop=True, inplace=True)

    print(f"[SOURCE COLD ATTACK] 🧊 Cold-start items (only 1 interaction): {len(cold_items)}")
    print(f"[SOURCE COLD ATTACK] 🧪 Successfully injected {len(new_rows)} fake interactions.")

    return attacked_df


def attack_all_items_weighted(df, alpha=2.0, user_ratio=0.15, inject_ratio=0.3):
    import random
    df = df.copy()
    item_counts = df["item"].value_counts()
    total_items = len(item_counts)

    sorted_items = item_counts.index.tolist()
    sorted_counts = item_counts.values.tolist()

    total_interactions = sum(sorted_counts)
    target_per_group = total_interactions / 10

    # === 分組 ===
    groups = []
    current_group = []
    current_sum = 0

    for item, count in zip(sorted_items, sorted_counts):
        current_group.append(item)
        current_sum += count
        if current_sum >= target_per_group:
            groups.append(current_group)
            current_group = []
            current_sum = 0

    if current_group:
        groups.append(current_group)

    while len(groups) < 10:
        groups.append([])

    # === 抽取 15% 攻擊使用者 ===
    all_users = list(df["user"].unique())
    attack_users = random.sample(all_users, int(len(all_users) * user_ratio))
    attack_users_set = set(attack_users)

    # === 計算注入上限 ===
    attack_user_interactions = df[df["user"].isin(attack_users)]
    total_attack_user_interactions = len(attack_user_interactions)
    max_total_injection = int(total_attack_user_interactions * inject_ratio)

    # === 權重分配策略 ===
    raw_weights = [alpha ** (9 - i) for i in range(10)]  # group_1 最多, group_10 最少
    total_weight = sum(raw_weights)
    normalized_weights = [w / total_weight for w in raw_weights]
    group_attack_counts = [int(max_total_injection * w) for w in normalized_weights]

    # === 注入資料 ===
    new_rows = []
    for idx, (group_items, group_quota) in enumerate(zip(groups, group_attack_counts)):
        copied = 0
        for item in group_items:
            current_users = set(df[df["item"] == item]["user"])
            candidate_users = list(attack_users_set - current_users)
            if not candidate_users or copied >= group_quota:
                continue

            random.shuffle(candidate_users)
            for user in candidate_users:
                if copied >= group_quota:
                    break
                user_interactions = df[df["user"] == user]
                if len(user_interactions) < 2:
                    continue
                second_last_time = user_interactions.iloc[-2]["timestamp"]
                inject_timestamp = second_last_time - 1e-4

                new_rows.append({
                    "user": user,
                    "item": item,
                    "timestamp": inject_timestamp,
                    "click": 1.0,
                    "is_target": 1
                })
                copied += 1

    # === 合併資料 ===
    attacked_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    attacked_df.sort_values(by="timestamp", inplace=True)
    attacked_df.reset_index(drop=True, inplace=True)

    # === 輸出統計 ===
    print(f"🎯 Selected attack users: {len(attack_users)} users")
    print(f"📊 Total interactions from attack users: {total_attack_user_interactions}")
    print(f"🔢 Max total injection count allowed: {max_total_injection}")
    print(f"🧪 Injected {len(new_rows)} fake interactions.")
    print(f"✅ Total items: {total_items}")
    for i, g in enumerate(groups):
        total_group_count = sum(item_counts[item] for item in g)
        print(f"📦 Group {i+1}: {len(g)} items, total count = {total_group_count}, "
              f"weight = {normalized_weights[i]:.4f}, quota = {group_attack_counts[i]}")

    return attacked_df



def attack_all_items_weighted_forward(
    df: pd.DataFrame,
    model: Model,
    source_edge_index: torch.Tensor,
    target_edge_index: torch.Tensor,
    num_users: int,
    num_target_items: int,
    *,
    alpha: float = 2.0,
    user_ratio: float = 0.15,
    inject_ratio: float = 0.30,
    device: str = "cuda:0"
) -> pd.DataFrame:
    """
    依 popularity 分組 + 權重注入，但使用 model.forward() 挑選「最低分」的 (user,item)

    Parameters
    ----------
    df : 已 re-index 的 target_df（含欄位 user, item, timestamp, click, is_target）
    model : 訓練好的 baseline 模型（已 .eval()）
    source_edge_index, target_edge_index : 2×E tensor
    num_users, num_target_items : 用於枚舉 candidate item
    alpha, user_ratio, inject_ratio : 與舊函式相同
    """
    df = df.copy()
    item_counts = df["item"].value_counts()
    total_items = len(item_counts)

    # ---------- popularity 分 10 組 ----------
    sorted_items  = item_counts.index.tolist()
    sorted_counts = item_counts.values.tolist()
    total_int     = item_counts.sum()
    tgt_per_grp   = total_int / 10

    groups, cur, cur_sum = [], [], 0
    for iid, cnt in zip(sorted_items, sorted_counts):
        cur.append(iid); cur_sum += cnt
        if cur_sum >= tgt_per_grp:
            groups.append(cur); cur, cur_sum = [], 0
    if cur: groups.append(cur)
    while len(groups) < 10: groups.append([])

    # ---------- 15% 攻擊使用者 ----------
    all_users        = list(df["user"].unique())
    attack_users     = random.sample(all_users, int(len(all_users)*user_ratio))
    attack_users_set = set(attack_users)

    attack_user_int  = df[df["user"].isin(attack_users)]
    max_total_inj    = int(len(attack_user_int) * inject_ratio)

    # ---------- 每組 quota ----------
    raw_w   = [alpha ** (9-i) for i in range(10)]
    norm_w  = [w/sum(raw_w) for w in raw_w]
    grp_q   = [int(max_total_inj * w) for w in norm_w]

    # ---------- 預先整理 user 已互動 item ----------
    user_seen = {
        u: set(df[df["user"] == u]["item"])
        for u in attack_users
    }

    # ---------- 注入 ----------
    new_rows = []
    model = model.to(device).eval()
    se = source_edge_index.to(device)
    te = target_edge_index.to(device)

    all_target_items = list(range(num_users, num_users+num_target_items))

    with torch.no_grad():
        for g_items, quota in zip(groups, grp_q):
            copied = 0
            for iid in g_items:
                if copied >= quota: break
                cand_users = [u for u in attack_users if iid not in user_seen[u]]
                if not cand_users: continue

                # batch 預測分數
                u_tensor = torch.tensor(cand_users, dtype=torch.long, device=device)
                i_tensor = torch.tensor([iid]*len(cand_users), dtype=torch.long, device=device)
                links    = torch.stack([u_tensor, i_tensor], dim=0)

                scores = model(se, te, links, is_source=False).squeeze()
                order  = torch.argsort(scores)              # 低 → 高

                for idx in order.cpu().tolist():
                    if copied >= quota: break
                    u = cand_users[idx]
                    hist = df[df["user"] == u]
                    if len(hist) < 2: continue
                    ts = hist.iloc[-2]["timestamp"] - 1e-4

                    new_rows.append({
                        "user": u,
                        "item": iid,
                        "timestamp": ts,
                        "click": 1.0,
                        "is_target": 1
                    })
                    copied += 1
                    user_seen[u].add(iid)            # 避免重複
            print(f"Group injected {copied}/{quota}")

    attacked_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    attacked_df.sort_values("timestamp", inplace=True)
    attacked_df.reset_index(drop=True, inplace=True)

    # ---------- 報表 ----------
    print(f"🎯 Selected attack users          : {len(attack_users)}")
    print(f"📊 Interactions of attack users   : {len(attack_user_int)}")
    print(f"🔢 Max total injection allowed    : {max_total_inj}")
    print(f"🧪 Injected fake interactions     : {len(new_rows)}")
    print(f"✅ Total items                    : {total_items}")
    for i, (g, q) in enumerate(zip(groups, grp_q)):
        cnt = sum(item_counts[item] for item in g)
        print(f"📦 Group {i+1}: {len(g)} items, total={cnt}, "
              f"weight={norm_w[i]:.4f}, quota={q}")

    return attacked_df





def attack_target_top1pct(df, top_pct: float = 0.01, user_pct: float = 0.05):
    """
    對 *target-domain* 進行資料注入：
    1. 先找出全資料 (source+target) 中前 top_pct 熱門的 item。
    2. 對每個熱門 item，從「沒買過此 item 的所有使用者」中，隨機挑選 user_pct 的人
       （向上取整，至少 1 人）來注入假互動。新互動標記為 is_target == 1。
    3. 回傳加上假互動後、依 timestamp 排序的完整 DataFrame。

    Parameters
    ----------
    df : pd.DataFrame
        必須含 ["user", "item", "timestamp", "click", "is_target"]。
    top_pct : float
        熱門 item 前百分比。預設 0.01 → 前 1 %。
    user_pct : float
        每個熱門 item 注入的使用者百分比。預設 0.05 → 5 %。

    Returns
    -------
    pd.DataFrame
        加入假互動後的完整資料（含 source & target）。
    """

    # ---------- 1. 樞紐統計：找出前 top_pct 熱門 item ----------
    item_counts = df["item"].value_counts()
    total_items = len(item_counts)
    top_k = max(1, ceil(total_items * top_pct))          # 至少 1 個
    top_items = item_counts.nlargest(top_k).index.tolist()

    print(f"[TOP-1% ATTACK] 🔥 熱門前 {top_pct*100:.0f}% item 數量 = {len(top_items)}")
    print("[TOP-1% ATTACK] 🏆 Top-10 item 互動數 =",
          item_counts.nlargest(10).tolist())

    # ---------- 2. 準備注入 ----------
    all_users = set(df["user"].unique())
    n_inject_users = max(1, ceil(len(all_users) * user_pct))

    new_rows = []

    for item in top_items:
        buyers = set(df.loc[df["item"] == item, "user"])
        candidate_users = list(all_users - buyers)
        if not candidate_users:   # 沒人可注入
            continue

        inject_users = random.sample(candidate_users,
                                     min(n_inject_users, len(candidate_users)))

        for user in inject_users:
            user_hist = df[df["user"] == user]
            if len(user_hist) < 2:
                # 與原邏輯一致：需至少 2 筆互動，才有「倒數第二筆」可參考
                continue

            second_last_ts = user_hist.iloc[-2]["timestamp"]
            fake_ts = second_last_ts - 1e-4

            new_rows.append({
                "user":      user,
                "item":      item,
                "timestamp": fake_ts,
                "click":     1.0,
                "is_target": 1           # ★ target-domain
            })

    # ---------- 3. 合併並輸出 ----------
    attacked_df = pd.concat([df, pd.DataFrame(new_rows)],
                            ignore_index=True)
    attacked_df.sort_values("timestamp", inplace=True)
    attacked_df.reset_index(drop=True, inplace=True)

    print(f"[TOP-1% ATTACK] 🧪 新增假互動筆數 = {len(new_rows)} "
          f"(理論上 ≈ {len(top_items)} × {n_inject_users})")

    return attacked_df


def attack_target_top1pct_model(
    df: pd.DataFrame,
    model,                                 # ⭐ 新增：已 .eval() 的模型
    source_edge_index: torch.Tensor,       # ⭐ 新增
    target_edge_index: torch.Tensor,       # ⭐ 新增
    *,
    top_pct: float = 0.01,                 # 熱門前 1 %
    low_pct: float = 0.01,                 # 取「最低 1 %」使用者
    chunk_size: int = 4096,                # GPU 記憶體不足時可調小
    device: str = "cuda:0"
) -> pd.DataFrame:
    """
    對 *target-domain* 進行假互動注入：
    1. 找出前 `top_pct` 熱門 item。
    2. 對每個熱門 item，計算所有「沒買過該 item」使用者的推薦分數，
       取分數最低的 `low_pct` 使用者注入 (is_target=1)。
    3. 回傳含原始 + 假互動的完整 DataFrame。
    """
    df = df.copy()

    # ---------- 熱門 item ----------
    item_counts = df["item"].value_counts()
    total_items = len(item_counts)
    top_k = max(1, ceil(total_items * top_pct))
    top_items = item_counts.nlargest(top_k).index.tolist()

    print(f"[TOP ATTACK] 🔥 Top {top_pct*100:.0f}% items = {len(top_items)}")
    print("[TOP ATTACK] 🏆 Top-10 item counts =",
          item_counts.nlargest(10).tolist())

    all_users = df["user"].unique().tolist()

    # ---------- 快速查表：使用者已互動的 item ----------
    user_seen = df.groupby("user")["item"].apply(set).to_dict()

    # ---------- 準備模型 ----------
    model = model.to(device).eval()
    se, te = source_edge_index.to(device), target_edge_index.to(device)

    def predict_scores(users, iid):
        """
        回傳 shape=[len(users)] 的分數 (越大 = 越被推薦)。
        依 chunk_size 拆 batch 以節省 GPU 記憶體。
        """
        scores = []
        for s in range(0, len(users), chunk_size):
            batch_users = users[s:s+chunk_size]
            u_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
            i_tensor = torch.full_like(u_tensor, iid)
            links    = torch.stack([u_tensor, i_tensor], dim=0)
            with torch.no_grad():
                batch = model(se, te, links, is_source=False).squeeze()
            scores.append(batch)
        return torch.cat(scores, dim=0)          # [N]

    # ---------- 注入 ----------
    new_rows = []
    for iid in top_items:
        # 1. 還沒買過該 item 的使用者
        cand_users = [u for u in all_users if iid not in user_seen.get(u, set())]
        if not cand_users:
            continue

        # 2. 預測推薦分數
        scores = predict_scores(cand_users, iid)

        # 3. 取最低 low_pct
        k = max(1, math.ceil(len(cand_users) * low_pct))
        tail_idx = torch.argsort(scores)[:k].cpu().tolist()
        tail_users = [cand_users[i] for i in tail_idx]

        # 4. 寫入假互動
        for u in tail_users:
            hist = df[df["user"] == u]
            if len(hist) < 2:              # 與舊邏輯一致
                continue
            ts = hist.iloc[-2]["timestamp"] - 1e-4
            new_rows.append({
                "user":      u,
                "item":      iid,
                "timestamp": ts,
                "click":     1.0,
                "is_target": 1
            })
            user_seen[u].add(iid)          # 避免重複

        print(f"  ⮑ item {iid}: injected {len(tail_users)}")

    # ---------- 合併並排序 ----------
    attacked_df = pd.concat([df, pd.DataFrame(new_rows)],
                            ignore_index=True).sort_values("timestamp")
    attacked_df.reset_index(drop=True, inplace=True)

    print(f"\n🧪 Total fake interactions injected = {len(new_rows)}")
    return attacked_df



def attack_target_top1pct_model_dbg(
    df: pd.DataFrame,
    model,
    source_edge_index: torch.Tensor,
    target_edge_index: torch.Tensor,
    *,
    top_pct: float = 0.01,     # 熱門前 1 %
    low_pct: float = 0.01,     # 取分數「最低 1 %」使用者
    chunk_size: int = 4096,
    device: str = "cuda:0"
) -> pd.DataFrame:
    df = df.copy()

    # ---------- 熱門 item ----------
    item_counts = df["item"].value_counts()
    total_items = len(item_counts)
    top_k = max(1, ceil(total_items * top_pct))
    top_items = item_counts.nlargest(top_k).index.tolist()

    print(f"[TOP ATTACK] 🔥 Top {top_pct*100:.0f}% items = {len(top_items)}")
    print("[TOP ATTACK] 🏆 Top-10 item counts =", item_counts.nlargest(10).tolist())

    all_users = df["user"].unique().tolist()
    user_seen = df.groupby("user")["item"].apply(set).to_dict()

    # ---------- 模型 ----------
    model = model.to(device).eval()
    se, te = source_edge_index.to(device), target_edge_index.to(device)

    def predict_scores(users, iid):
        scores = []
        for s in range(0, len(users), chunk_size):
            batch_users = users[s:s+chunk_size]
            u_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
            i_tensor = torch.full_like(u_tensor, iid)
            links    = torch.stack([u_tensor, i_tensor], dim=0)
            with torch.no_grad():
                batch = model(se, te, links, is_source=False).squeeze()
            scores.append(batch)
        return torch.cat(scores, 0)  # [N]

    # ---------- 注入 ----------
    new_rows = []
    for iid in top_items:
        cand_users = [u for u in all_users if iid not in user_seen.get(u, set())]
        if not cand_users:
            continue

        scores = predict_scores(cand_users, iid)
        k = max(1, math.ceil(len(cand_users) * low_pct))
        tail_idx = torch.argsort(scores)[:k].cpu()
        tail_users  = [cand_users[i] for i in tail_idx]
        tail_scores = scores[tail_idx].cpu().tolist()

        # ★★★ DEBUG: 列印最低 1 % 使用者及其分數 ★★★
        print(f"\n[item {iid}] lowest {low_pct*100:.1f}% users ({k}人):")
        for u, sc in zip(tail_users, tail_scores):
            print(f"  user {u:<8} score={sc:.6f}")

        # 寫入假互動
        for u in tail_users:
            hist = df[df["user"] == u]
            if len(hist) < 2:
                continue
            ts = hist.iloc[-2]["timestamp"] - 1e-4
            new_rows.append({
                "user":      u,
                "item":      iid,
                "timestamp": ts,
                "click":     1.0,
                "is_target": 1
            })
            user_seen[u].add(iid)

        print(f"  ⮑ injected {len(tail_users)} interactions.")

    # ---------- 合併 ----------
    attacked_df = (
        pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
          .sort_values("timestamp")
          .reset_index(drop=True)
    )

    print(f"\n🧪 Total fake interactions injected = {len(new_rows)}")
    return attacked_df

def attack_top1pct_fixed_users(
    df: pd.DataFrame,
    model,
    source_edge_index: torch.Tensor,
    target_edge_index: torch.Tensor,
    *,
    top_pct: float = 0.01,
    low_pct: float = 0.03,
    chunk_size: int = 1024,
    device: str = "cuda:0"
) -> pd.DataFrame:
    df = df.copy()

    # ---------- 挑 top 1% item ----------
    item_counts = df["item"].value_counts()
    top_k = max(1, math.ceil(len(item_counts) * top_pct))
    top_items = item_counts.nlargest(top_k).index.tolist()
    print(f"[TOP ATTACK] 🔥 Top {top_pct*100:.0f}% items = {len(top_items)}")

    all_users = df["user"].unique().tolist()
    num_users = len(all_users)
    num_items = len(top_items)

    model = model.to(device).eval()
    se, te = source_edge_index.to(device), target_edge_index.to(device)

    # ---------- 計算所有 user 對 top_items 的平均分數 ----------
    user_scores = torch.zeros(num_users, device=device)
    for iid in top_items:
        item_scores = []
        for s in range(0, num_users, chunk_size):
            batch_users = all_users[s:s+chunk_size]
            u_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
            i_tensor = torch.full_like(u_tensor, iid)
            links = torch.stack([u_tensor, i_tensor], dim=0)
            with torch.no_grad():
                batch = model(se, te, links, is_source=False).squeeze()
            item_scores.append(batch)
        item_scores = torch.cat(item_scores, 0)  # [num_users]
        user_scores += item_scores  # 累加分數

    user_scores /= num_items  # 平均分數

    # ---------- 選最低分 1% user ----------
    low_k = max(1, math.ceil(num_users * low_pct))
    sorted_idx = torch.argsort(user_scores)[:low_k]
    lowest_users = [all_users[i] for i in sorted_idx.cpu()]
    print(f"[TOP ATTACK] 🧨 Lowest {low_pct*100:.1f}% users = {len(lowest_users)}")

    # ---------- 對這些 user 注入所有 top_items ----------
    new_rows = []
    unsafe_users = []

    for u in lowest_users:
        hist = df[df["user"] == u].sort_values("timestamp")

        # 確認倒數第二筆 timestamp
        if len(hist) >= 2:
            ts_safe_limit = hist.iloc[-2]["timestamp"]
            ts_base = ts_safe_limit - 1e-4
        elif len(hist) == 1:
            # 只有一筆，test 就是這筆，要保證在之前
            ts_safe_limit = hist.iloc[-1]["timestamp"]
            ts_base = ts_safe_limit - 1e-4
        else:
            # 完全沒紀錄，給一個極早時間
            ts_safe_limit = float("inf")  # 不影響
            ts_base = 0.0

        # **安全檢查：確保注入的時間 < 倒數第二筆**
        if ts_base >= ts_safe_limit:
            print(f"[WARN] User {u} ts_base >= ts_safe_limit?! Forcing earlier timestamp.")
            ts_base = ts_safe_limit - 1e-4

        # 注入所有 top_items
        for iid in top_items:
            new_rows.append({
                "user":      u,
                "item":      iid,
                "timestamp": ts_base,
                "click":     1.0,
                "is_target": 1
            })

    attacked_df = (
        pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
          .sort_values("timestamp")
          .reset_index(drop=True)
    )

    # ---------- Debug 檢查：確認沒有污染最後兩筆 ----------
    print("\n[DEBUG] ✅ Checking last two timestamps for each user...")
    touched_test = []
    for u in lowest_users:
        orig_hist = df[df["user"] == u].sort_values("timestamp")
        new_hist  = attacked_df[attacked_df["user"] == u].sort_values("timestamp")

        if len(orig_hist) >= 2:
            orig_val_ts  = orig_hist.iloc[-2]["timestamp"]
            orig_test_ts = orig_hist.iloc[-1]["timestamp"]

            new_val_ts  = new_hist.iloc[-2]["timestamp"]
            new_test_ts = new_hist.iloc[-1]["timestamp"]

            # 確保最後兩筆沒變
            if not (math.isclose(orig_val_ts, new_val_ts) and math.isclose(orig_test_ts, new_test_ts)):
                touched_test.append(u)

    if len(touched_test) == 0:
        print("[DEBUG] ✅ No test/val data was touched. Safe!")
    else:
        print(f"[DEBUG] ❌ WARNING: {len(touched_test)} users' test/val were shifted!")
        print("Affected users:", touched_test[:10])

    print(f"\n🧪 Total fake interactions injected = {len(new_rows)} (expected {len(top_items) * len(lowest_users)})")
    return attacked_df
