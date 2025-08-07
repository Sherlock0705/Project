import pandas as pd
import logging

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

