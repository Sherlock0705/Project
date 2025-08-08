import pandas as pd
import logging
import torch, math
import random
import logging
import pandas as pd
from model import Model 
from math import ceil
def analyze_user_top_item_coverage(   #用來統計top-k%的item有多少user買過 多少user沒買過
    df: pd.DataFrame,
    k_percent: float = 0.01,
    user_col: str = "user",
    item_col: str = "item"
):
    """
    統計有多少 user 曾經購買過前 k% 熱門 item，以及完全沒買過的 user 數量。

    Args:
        df (pd.DataFrame): 包含 user 和 item 欄位的資料（通常是 target_df）
        k_percent (float): 熱門 item 的前百分比（如 0.01 表前 1%）
        user_col (str): user 欄位名稱
        item_col (str): item 欄位名稱

    Returns:
        dict: {
            "top_items": set,
            "user_with_top_item": set,
            "user_without_top_item": set
        }
    """
    top_k = int(df[item_col].nunique() * k_percent)
    top_items = set(df[item_col].value_counts().nlargest(top_k).index)

    logging.info(f"[前{k_percent*100:.2f}% 熱門 item] 數量: {len(top_items)}")

    user_with_top_item = set()
    user_all = set(df[user_col].unique())

    for user, group in df.groupby(user_col):
        items = set(group[item_col].values)
        if items & top_items:
            user_with_top_item.add(user)

    user_without_top_item = user_all - user_with_top_item

    logging.info(f"[熱門 item] 有買過的 user 數量: {len(user_with_top_item)}")
    logging.info(f"[熱門 item] 完全沒買過的 user 數量: {len(user_without_top_item)}")

    return {
        "top_items": top_items,
        "user_with_top_item": user_with_top_item,
        "user_without_top_item": user_without_top_item,
    }


def attack_target_top1pct(df, top_pct, hard_users):
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
    n_inject_users = max(1, ceil(len(hard_users) * 1))
    hard_users = set(hard_users)  
    new_rows = []

    for item in top_items:
        buyers = set(df.loc[df["item"] == item, "user"])
        candidate_users = list(hard_users - buyers)
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
