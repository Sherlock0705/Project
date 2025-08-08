import pandas as pd
import random
import math
import logging
import os
import numpy as np
import math
import pandas as pd
import random
import torch   # ← 加在這裡
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

    

import math
import random
import pandas as pd

def attack_clone_popular_users(df, top_item_pct=0.03, user_sample_pct=0.03, seed=42):
    """
    攻擊流程（修正版）：
    1. 只看目標域 (is_target==1)，計算 item 熱度，取前 top_item_pct 最熱門 Target 商品。
    2. 鎖定來源域 (is_target==0)，找出「沒有買過這些熱門 Target 商品」的使用者。
    3. 隨機抽 user_sample_pct 比例的使用者 (>=1)。
       以抽樣列表第一位為「模板」：把他在來源域的所有購買紀錄
       複製給抽樣 list 內其餘使用者（覆蓋其原來源域紀錄）。
    4. 回傳新的 attacked_df。

    參數
    ----
    df : pd.DataFrame，欄位需含 ["user","item","timestamp","click","is_target"]
    top_item_pct : float，取熱門 Target 商品百分比 (預設10%)
    user_sample_pct : float，抽樣沒買過熱門 Target 商品的來源用戶百分比 (預設50%)
    seed : int，隨機種子
    """
    random.seed(seed)
    df = df.copy()
    
    # 切分來源/目標域
    src_df = df[df["is_target"] == 0].copy()
    tgt_df = df[df["is_target"] == 1].copy()

    # ① 在目標域找「前 top_item_pct 熱門商品」
    tgt_item_cnt = tgt_df["item"].value_counts()
    top_k = max(1, math.ceil(len(tgt_item_cnt) * top_item_pct))
    top_target_items = set(tgt_item_cnt.iloc[:top_k].index)

    # ② 找來源域中「沒有買過這些熱門 Target 商品」的使用者
    src_users = src_df["user"].unique().tolist()
    buyers_of_top_target = (
        df[df["item"].isin(top_target_items)]["user"].unique().tolist()
    )
    non_buyers = [u for u in src_users if u not in buyers_of_top_target]

    if not non_buyers:
        print("[CLONE ATTACK] ❗ 來源域中沒有符合條件（未買過熱門 Target 商品）的使用者，結束")
        return df

    # ③ 隨機抽樣
    n_sample = max(1, math.ceil(len(non_buyers) * user_sample_pct))
    sample_users = random.sample(non_buyers, n_sample)

    # 模板使用者
    template_user = sample_users[0]
    template_records = src_df[src_df["user"] == template_user].copy()
    if template_records.empty:
        print("[CLONE ATTACK] ❗ 模板使用者在來源域無紀錄，結束")
        return df

    # ④ 刪除抽樣用戶原有的來源域行為
    src_df = src_df[~src_df["user"].isin(sample_users)].copy()

    # ⑤ 生成複製紀錄
    clones = []
    for u in sample_users:
        tmp = template_records.copy()
        tmp["user"] = u
        clones.append(tmp)

    attacked_src = pd.concat([src_df] + clones, ignore_index=True)
    attacked_src.sort_values(["user", "timestamp"], inplace=True)
    attacked_src.reset_index(drop=True, inplace=True)

    # ⑥ 與目標域資料再合併
    attacked_df = pd.concat([attacked_src, tgt_df], ignore_index=True)
    attacked_df.sort_values("timestamp", inplace=True)
    attacked_df.reset_index(drop=True, inplace=True)


    # ⑦ Log
    print(f"[CLONE ATTACK] 🔥 目標域熱門商品前 {top_item_pct*100:.0f}% = {top_k} 件")
    print(f"[CLONE ATTACK] 👥 來源域用戶總數 = {len(src_users)}")
    print(f"[CLONE ATTACK] 🚫 未買過熱門 Target 商品的來源用戶 = {len(non_buyers)}")
    print(f"[CLONE ATTACK] 🎯 抽樣 {n_sample} 位用戶 (首位作模板)")
    print(f"[CLONE ATTACK] 🧪 複製紀錄行數 = {template_records.shape[0]*(n_sample-1)}")

    return attacked_df





def attack_clone_popular_users_2(src_df, tgt_df, top_item_pct=0.03, user_modify_pct=0.03, seed=None):
    """
    只針對沒買過熱門商品的用戶，隨機選 user_modify_pct 當 receiver，
    每個 receiver 從一個隨機 giver 複製一半來源域紀錄，receiver 只被更改一次。
    其他邏輯同原本。
    """
    import random
    import math
    import pandas as pd

    random.seed(seed)

    print("src_df shape:", src_df.shape)
    print("tgt_df shape:", tgt_df.shape)
    print(src_df.head())
    print(tgt_df.head())

    # 熱門 item（以 tgt_df 為主）
    item_cnt = tgt_df["item"].value_counts()
    top_k = max(1, math.ceil(len(item_cnt) * top_item_pct))
    popular_items = set(item_cnt.iloc[:top_k].index)

    # 買過熱門商品的用戶（target domain）
    buyers = tgt_df[tgt_df["item"].isin(popular_items)]["user"].unique().tolist()
    # 沒買過熱門商品的用戶（target domain）
    all_users = tgt_df["user"].unique().tolist()
    non_buyers = list(set(all_users) - set(buyers))

    if not buyers or not non_buyers:
        print("[CLONE RAND] 買過或沒買過熱門商品的人數不足，結束")
        return src_df

    # 隨機選 receiver（沒買過熱門商品的 user）
    n_modify = max(1, math.ceil(len(all_users) * user_modify_pct))
    sample_receivers = random.sample(buyers, min(len(buyers), n_modify))

    new_rows = []
    for receiver in sample_receivers:
        # 隨機選 giver（有買過熱門商品，且不是自己）
        valid_givers = [u for u in buyers if u != receiver]
        if not valid_givers:
            continue
        giver = random.choice(valid_givers)
        giver_records = src_df[src_df["user"] == giver]
        if giver_records.empty:
            continue
        n_half = max(1, len(giver_records) // 2)
        sampled_records = giver_records.iloc[:n_half].copy()
        sampled_records["user"] = receiver
        new_rows.append(sampled_records)

    # 合併新紀錄
    attacked_src = pd.concat([src_df] + new_rows, ignore_index=True)
    attacked_src.sort_values(["user", "timestamp"], inplace=True)
    attacked_src.reset_index(drop=True, inplace=True)

    print(f"[CLONE RAND] 🎯 被更改顧客數: {len(new_rows)}")
    print(f"[CLONE RAND] 🧪 新增紀錄行數: {sum(len(x) for x in new_rows)}")
    return attacked_src

def attack_clone_popular_users_3(src_df, tgt_df, top_item_pct=0.03, user_modify_pct=0.03, seed=None):
    """
    receiver 一半為有買過熱門商品的 user，另一半為沒買過熱門商品的 user，
    每個 receiver 從一個隨機 giver 複製一半來源域紀錄，receiver 只被更改一次。
    其他邏輯同原本。
    """
    import random
    import math
    import pandas as pd

    random.seed(seed)

    # 熱門 item（以 tgt_df 為主）
    item_cnt = tgt_df["item"].value_counts()
    top_k = max(1, math.ceil(len(item_cnt) * top_item_pct))
    popular_items = set(item_cnt.iloc[:top_k].index)

    # 有買過／沒買過熱門商品的 user
    buyers = tgt_df[tgt_df["item"].isin(popular_items)]["user"].unique().tolist()
    all_users = tgt_df["user"].unique().tolist()
    non_buyers = list(set(all_users) - set(buyers))

    if not buyers or not non_buyers:
        print("[CLONE RAND] 買過或沒買過熱門商品的人數不足，結束")
        return src_df

    # 計算要抽的數量（總數 user_modify_pct * user 數量）
    n_modify = max(1, math.ceil(len(all_users) * user_modify_pct))
    n_buyers = min(len(buyers), (n_modify + 1) // 2)         # 向上取整
    n_non_buyers = min(len(non_buyers), n_modify // 2)       # 向下取整

    sample_buyers = random.sample(buyers, n_buyers)
    sample_non_buyers = random.sample(non_buyers, n_non_buyers)
    sample_receivers = sample_buyers + sample_non_buyers

    new_rows = []
    for receiver in sample_receivers:
        # giver 不能是 receiver 本人
        valid_givers = [u for u in buyers if u != receiver]
        if not valid_givers:
            continue
        giver = random.choice(valid_givers)
        giver_records = src_df[src_df["user"] == giver]
        if giver_records.empty:
            continue
        n_half = max(1, len(giver_records) // 2)
        sampled_records = giver_records.iloc[:n_half].copy()
        sampled_records["user"] = receiver
        new_rows.append(sampled_records)

    # 合併新紀錄
    attacked_src = pd.concat([src_df] + new_rows, ignore_index=True)
    attacked_src.sort_values(["user", "timestamp"], inplace=True)
    attacked_src.reset_index(drop=True, inplace=True)

    print(f"[CLONE RAND] 🎯 被更改顧客數: {len(new_rows)}")
    print(f"[CLONE RAND] 🧪 新增紀錄行數: {sum(len(x) for x in new_rows)}")
    return attacked_src
def get_users_with_lowest_avg_scores(baseline_model, src_df, top_items, pct=0.01, device="cuda:1",source_edge_index=None,target_edge_index=None):
    user_ids = src_df["user"].unique()
    user_id_map = {u: i for i, u in enumerate(user_ids)}
    item_id_map = {it: i for i, it in enumerate(top_items)}
    
    user_scores = []
    baseline_model.eval()
    with torch.no_grad():
        for u in user_ids:
            user_index = user_id_map[u]
            # item indices 以source domain index為主
            item_indices = [item_id_map[it] for it in top_items if it in item_id_map]
            if not item_indices:
                continue
            user_tensor = torch.tensor([user_index] * len(item_indices), device=device)
            item_tensor = torch.tensor(item_indices, device=device)
            link = torch.stack([user_tensor, item_tensor], dim=0)
            # 根據你的forward: source_edge_index, target_edge_index, link, is_source
            # 這裡假設你有edge_index可用（給個dummy，或放global變數）
            # 若你的forward要求edge_index，建議先傳入全圖 source_edge_index, target_edge_index
            scores = baseline_model(
                source_edge_index, target_edge_index, link, is_source=True
            ).view(-1)
            avg_score = scores.mean().item()
            user_scores.append((u, avg_score))
    user_scores.sort(key=lambda x: x[1])  # 由小到大
    n_lowest = max(1, int(len(user_ids) * pct))
    lowest_users = [u for u, s in user_scores[:n_lowest]]
    return lowest_users
def attack_clone_popular_users_low_score(
    src_df, tgt_df, baseline_model,
    source_edge_index, target_edge_index,
    top_item_pct=0.03, user_modify_pct=0.01, seed=None, device="cuda:1"
):
    """
    使用 model 推薦分數，選出 top 3% item、找出推薦度最低的 1% user 作為 receiver，
    每個 receiver 從隨機 giver 複製一半來源域紀錄。
    """
    import random
    import math
    import pandas as pd
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    
    # 熱門 item（以 tgt_df 為主）
    item_cnt = tgt_df["item"].value_counts()
    top_k = max(1, math.ceil(len(item_cnt) * top_item_pct))
    popular_items = list(item_cnt.iloc[:top_k].index)
    
    # 挑出 baseline_model 預測 top 3% item 平均分數最低的 1% user
    lowest_users = get_users_with_lowest_avg_scores(
        baseline_model, src_df, popular_items, pct=user_modify_pct, device=device,source_edge_index=source_edge_index, target_edge_index=target_edge_index
    )
    
    # 挑一組 giver（買過熱門商品的 user，且不是 receiver 本人）
    buyers = tgt_df[tgt_df["item"].isin(popular_items)]["user"].unique().tolist()

    new_rows = []
    for receiver in lowest_users:
        valid_givers = [u for u in buyers if u != receiver]
        if not valid_givers:
            continue
        giver = random.choice(valid_givers)
        giver_records = src_df[src_df["user"] == giver]
        if giver_records.empty:
            continue
        n_half = max(1, len(giver_records) // 2)
        sampled_records = giver_records.iloc[:n_half].copy()
        sampled_records["user"] = receiver
        new_rows.append(sampled_records)

    # 合併新紀錄
    attacked_src = pd.concat([src_df] + new_rows, ignore_index=True)
    attacked_src.sort_values(["user", "timestamp"], inplace=True)
    attacked_src.reset_index(drop=True, inplace=True)

    print(f"[CLONE ATTACK] 🎯 被更改顧客數: {len(new_rows)}")
    print(f"[CLONE ATTACK] 🧪 新增紀錄行數: {sum(len(x) for x in new_rows)}")
    return attacked_src

def attack_clone_popular_users_low_score_target(
    src_df, tgt_df, baseline_model,
    source_edge_index, target_edge_index,
    top_item_pct=0.03, user_modify_pct=0.01, seed=None, device="cuda:1",
    remove_last_popular_pct=0.00
):
    import random
    import math
    import pandas as pd
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

    # ===== 隔離每個 user 最後兩筆（valid/test）=====
    last_two_idx = []
    other_idx = []
    for user, group in tgt_df.groupby("user"):
        if len(group) <= 2:
            last_two_idx.extend(group.index)
        else:
            last_two_idx.extend(group.index[-2:])
            other_idx.extend(group.index[:-2])
    last_two_df = tgt_df.loc[last_two_idx].copy()  # valid/test
    other_df = tgt_df.loc[other_idx].copy()        # 可被 attack

    # 熱門 item（以可被 attack 部分為主）
    item_cnt = other_df["item"].value_counts()
    top_k = max(1, math.ceil(len(item_cnt) * top_item_pct))
    popular_items = list(item_cnt.iloc[:top_k].index)

    # 推薦分數最低的 1% user
    lowest_users = get_users_with_lowest_avg_scores(
        baseline_model, src_df, popular_items, pct=user_modify_pct,
        device=device, source_edge_index=source_edge_index, target_edge_index=target_edge_index
    )
    buyers = other_df[other_df["item"].isin(popular_items)]["user"].unique().tolist()

    new_rows = []
    for receiver in lowest_users:
        valid_givers = [u for u in buyers if u != receiver]
        if not valid_givers:
            continue
        giver = random.choice(valid_givers)
        giver_records = other_df[other_df["user"] == giver]
        if giver_records.empty:
            continue
        n_half = max(1, len(giver_records) // 2)
        sampled_records = giver_records.iloc[:n_half].copy()
        sampled_records["user"] = receiver
        new_rows.append(sampled_records)

    # 合併新紀錄
    attacked_tgt = pd.concat([other_df] + new_rows, ignore_index=True)
    attacked_tgt.sort_values(["user", "timestamp"], inplace=True)
    attacked_tgt.reset_index(drop=True, inplace=True)

    # 刪除最後3%最熱門商品
    if remove_last_popular_pct > 0:
        item_counts = attacked_tgt["item"].value_counts()
        n_last3pct = max(1, int(len(item_counts) * remove_last_popular_pct))
        hottest_items = item_counts.iloc[-n_last3pct:].index.tolist()
        attacked_tgt = attacked_tgt[~attacked_tgt["item"].isin(hottest_items)]
        attacked_tgt.reset_index(drop=True, inplace=True)

    # 把 valid/test 回收回來
    final_tgt = pd.concat([attacked_tgt, last_two_df], ignore_index=True)
    final_tgt.sort_values(["user", "timestamp"], inplace=True)
    final_tgt.reset_index(drop=True, inplace=True)

    print(f"[CLONE ATTACK-TARGET] 🎯 被更改顧客數: {len(new_rows)}")
    print(f"[CLONE ATTACK-TARGET] 🧪 新增紀錄行數: {sum(len(x) for x in new_rows)}")
    return final_tgt


def find_similarity(target_df, source_df):
    # 1. 找出 target domain 熱門商品
    target_item_counts = target_df["item"].value_counts()
    target_top_k = int(len(target_item_counts) * 0.03)
    target_popular_items = set(target_item_counts.iloc[:target_top_k].index)
    target_unpopular_items = set(target_item_counts.iloc[int(len(target_item_counts) * 0.03):].index)

    # 2. 找出 source domain 熱門商品
    source_item_counts = source_df["item"].value_counts()
    source_top_k = int(len(source_item_counts) * 0.03)
    source_popular_items = set(source_item_counts.iloc[:source_top_k].index)
    source_unpopular_items = set(source_item_counts.iloc[int(len(source_item_counts) * 0.03):].index)

    # 3. 找出有購買 target 熱門商品的 user
    target_popular_users = set(
        target_df[target_df["item"].isin(target_popular_items)]["user"].unique()
    )

    # 4. 有購買 source 熱門商品的 user
    source_popular_users = set(
        source_df[source_df["item"].isin(source_popular_items)]["user"].unique()
    )

    # 5. 有購買 source 冷門商品的 user
    source_unpopular_users = set(
        source_df[source_df["item"].isin(source_unpopular_items)]["user"].unique()
    )

    # 6. 有購買 target 冷門商品的 user
    target_unpopular_users = set(
        target_df[target_df["item"].isin(target_unpopular_items)]["user"].unique()
    )

    # === A. target 熱門 ➜ source 熱門
    source_popular_user_df = source_df[
        (source_df["user"].isin(target_popular_users)) &
        (source_df["item"].isin(source_popular_items))
    ]
    users_in_both = set(source_popular_user_df["user"].unique())
    ratio1 = len(users_in_both) / len(target_popular_users) if target_popular_users else float('nan')

    # === B. target 熱門 ➜ source 冷門
    source_unpopular_user_df = source_df[
        (source_df["user"].isin(target_popular_users)) &
        (source_df["item"].isin(source_unpopular_items))
    ]
    users_in_both_cold = set(source_unpopular_user_df["user"].unique())
    ratio2 = len(users_in_both_cold) / len(target_popular_users) if target_popular_users else float('nan')

    # === C. source 熱門 ➜ target 冷門
    target_unpopular_user_df = target_df[
        (target_df["user"].isin(source_popular_users)) &
        (target_df["item"].isin(target_unpopular_items))
    ]
    users_in_cross = set(target_unpopular_user_df["user"].unique())
    ratio3 = len(users_in_cross) / len(source_popular_users) if source_popular_users else float('nan')

    # === 印出結果
    print(f"① 買了 target 熱門商品的 user，有 {ratio1:.2%} 也買了 source 熱門商品")
    print(f"② 買了 target 熱門商品的 user，有 {ratio2:.2%} 也買了 source 冷門商品")
    print(f"③ 買了 source 熱門商品的 user，有 {ratio3:.2%} 也買了 target 冷門商品")
