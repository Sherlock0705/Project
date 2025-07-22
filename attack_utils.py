import pandas as pd
import random

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

    multipliers = [0.15 - 0.01 * i for i in range(10)]

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
