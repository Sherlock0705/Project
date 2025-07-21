import logging

import torch
import torch.nn as nn
import wandb
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from auxilearn.optim import MetaOptimizer
from dataset import Dataset
from pytorchtools import EarlyStopping
from utils import link_split, load_model


def meta_optimizeation(
    target_meta_loader,
    replace_optimizer,
    model,
    args,
    criterion,
    replace_scheduler,
    source_edge_index,
    target_edge_index,
):
    device = args.device
    for batch, (target_link, target_label) in enumerate(target_meta_loader):
        if batch < args.descent_step:
            target_link, target_label = target_link.to(device), target_label.to(device)

            replace_optimizer.zero_grad()
            out = model.meta_prediction(
                source_edge_index, target_edge_index, target_link
            ).squeeze()
            loss_target = criterion(out, target_label).mean()
            loss_target.backward()
            replace_optimizer.step()
        else:
            break
    replace_scheduler.step()


@torch.no_grad()
def evaluate(name, model, source_edge_index, target_edge_index, link, label):
    model.eval()

    out = model(source_edge_index, target_edge_index, link, is_source=False).squeeze()
    try:
        auc = roc_auc_score(label.tolist(), out.tolist())
    except:
        auc = 1.0
    logging.info(f"{name} AUC: {auc:4f}")

    model.train()
    return auc


@torch.no_grad()
def evaluate_hr_at_k(model, source_edge_index, target_edge_index, data, k=10, num_negatives=99):
    model.eval()
    logging.info(f"Evaluating HR@{k} ...")

    user_pos_items = dict()
    for i in range(data.target_link.shape[1]):
        u, v = data.target_link[:, i].tolist()
        if data.split_mask["test"][i]:  # 只挑 test mask 的互動
            user_pos_items[u] = v  # 每位 user 最後一個互動

    all_items = set(range(data.num_users, data.num_users + data.num_target_items))
    hr_total = 0

    for user, pos_item in user_pos_items.items():
        # 負樣本：挑沒點過的 item（排除 pos_item）
        neg_items = list(all_items - set(data.target_link[1][data.target_link[0] == user].tolist()))
        if pos_item in neg_items:
            neg_items.remove(pos_item)

        if num_negatives is not None:
            sampled_neg_items = np.random.choice(neg_items, size=min(num_negatives, len(neg_items)), replace=False).tolist()
        else:
            sampled_neg_items = neg_items
        test_items = sampled_neg_items + [pos_item]
        device = next(model.parameters()).device
        test_links = torch.tensor([[user]*len(test_items), test_items], dtype=torch.long).to(device)


        scores = model(source_edge_index, target_edge_index, test_links, is_source=False).squeeze()
        _, indices = torch.topk(scores, k)

        topk_items = torch.tensor(test_items)[indices.cpu()]
        hit = int(pos_item in topk_items.tolist())
        hr_total += hit

    hr_at_k = hr_total / len(user_pos_items)
    logging.info(f"HR@{k}: {hr_at_k:.4f}")
    return hr_at_k

@torch.no_grad()
def evaluate_ndcg_at_k(model, source_edge_index, target_edge_index, data, k=10, num_negatives=99):
    model.eval()
    logging.info(f"Evaluating NDCG@{k} ...")

    user_pos_items = dict()
    for i in range(data.target_link.shape[1]):
        u, v = data.target_link[:, i].tolist()
        if data.split_mask["test"][i]:
            user_pos_items[u] = v  # 每位 user 最後一個互動

    all_items = set(range(data.num_users, data.num_users + data.num_target_items))
    ndcg_total = 0

    for user, pos_item in user_pos_items.items():
        neg_items = list(all_items - set(data.target_link[1][data.target_link[0] == user].tolist()))
        if pos_item in neg_items:
            neg_items.remove(pos_item)

        if num_negatives is not None:
            sampled_neg_items = np.random.choice(neg_items, size=min(num_negatives, len(neg_items)), replace=False).tolist()
        else:
            sampled_neg_items = neg_items
        test_items = sampled_neg_items + [pos_item]
        labels = [0] * len(sampled_neg_items) + [1]

        device = next(model.parameters()).device
        test_links = torch.tensor([[user]*len(test_items), test_items], dtype=torch.long).to(device)
        scores = model(source_edge_index, target_edge_index, test_links, is_source=False).squeeze()
        _, indices = torch.topk(scores, k)

        ranked_labels = torch.tensor(labels)[indices.cpu()].tolist()

        # 計算 DCG@k
        dcg = 0.0
        for idx, rel in enumerate(ranked_labels):
            if rel:
                dcg += 1.0 / np.log2(idx + 2)  # idx從0開始 +2

        # 計算 IDCG@k
        idcg = 1.0  # 因為只有一個正樣本，最佳排序時會在第一位
        ndcg = dcg / idcg
        ndcg_total += ndcg

    ndcg_at_k = ndcg_total / len(user_pos_items)
    logging.info(f"NDCG@{k}: {ndcg_at_k:.4f}")
    return ndcg_at_k


def train(model, perceptor, data, args):
    device = args.device
    data = data.to(device)
    model = model.to(device)
    perceptor = perceptor.to(device)

    (
        source_edge_index,
        source_label,
        source_link,
        target_train_edge_index,
        target_train_label,
        target_train_link,
        target_valid_link,
        target_valid_label,
        target_test_link,
        target_test_label,
    ) = link_split(data)

    source_set_size = source_link.shape[1]
    train_set_size = target_train_link.shape[1]
    val_set_size = target_valid_link.shape[1]
    test_set_size = target_test_link.shape[1]
    logging.info(f"Train set size: {train_set_size}")
    logging.info(f"Valid set size: {val_set_size}")
    logging.info(f"Test set size: {test_set_size}")

    target_train_set = Dataset(
        target_train_link.to("cpu"),
        target_train_label.to("cpu"),
    )
    target_train_loader = DataLoader(
        target_train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=target_train_set.collate_fn,
    )

    source_batch_size = int(args.batch_size * train_set_size / source_set_size)
    source_train_set = Dataset(source_link.to("cpu"), source_label.to("cpu"))
    source_train_loader = DataLoader(
        source_train_set,
        batch_size=source_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=source_train_set.collate_fn,
    )

    target_meta_loader = DataLoader(
        target_train_set,
        batch_size=args.meta_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=target_train_set.collate_fn,
    )
    target_meta_iter = iter(target_meta_loader)
    source_meta_batch_size = int(
        args.meta_batch_size * train_set_size / source_set_size
    )
    source_meta_loader = DataLoader(
        source_train_set,
        batch_size=source_meta_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=source_train_set.collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    perceptor_optimizer = torch.optim.Adam(
        perceptor.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    meta_optimizer = MetaOptimizer(
        meta_optimizer=perceptor_optimizer,
        hpo_lr=args.hpo_lr,
        truncate_iter=3,
        max_grad_norm=10,
    )

    model_param = [
        param for name, param in model.named_parameters() if "preds" not in name
    ]
    replace_param = [
        param for name, param in model.named_parameters() if name.startswith("replace")
    ]
    replace_optimizer = torch.optim.Adam(replace_param, lr=args.lr)
    replace_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        replace_optimizer, T_max=args.T_max
    )

    early_stopping = EarlyStopping(
        patience=args.patience,
        verbose=True,
        path=args.model_path,
        trace_func=logging.info,
    )

    criterion = nn.BCELoss(reduction="none")
    iteration = 0
    for epoch in range(args.epochs):
        for (source_link, source_label), (target_link, target_label) in zip(
            source_train_loader, target_train_loader
        ):
            torch.cuda.empty_cache()
            source_link = source_link.to(device)
            source_label = source_label.to(device)
            target_link = target_link.to(device)
            target_label = target_label.to(device)
            weight_source = perceptor(source_link[1], source_edge_index, model)

            optimizer.zero_grad()
            source_out = model(
                source_edge_index, target_train_edge_index, source_link, is_source=True
            ).squeeze()
            target_out = model(
                source_edge_index, target_train_edge_index, target_link, is_source=False
            ).squeeze()
            source_loss = (
                criterion(source_out, source_label).reshape(-1, 1) * weight_source
            ).sum()
            target_loss = criterion(target_out, target_label).mean()
            loss = source_loss + target_loss if args.use_meta else target_loss
            loss.backward()
            optimizer.step()

            iteration += 1
            if (
                args.use_source
                and args.use_meta
                and iteration % args.meta_interval == 0
            ):
                logging.info(f"Entering meta optimization, iteration: {iteration}")
                meta_optimizeation(
                    target_meta_loader,
                    replace_optimizer,
                    model,
                    args,
                    criterion,
                    replace_scheduler,
                    source_edge_index,
                    target_train_edge_index,
                )

                try:
                    target_meta_link, target_meta_label = next(target_meta_iter)
                except StopIteration:
                    target_meta_iter = iter(target_meta_loader)
                    target_meta_link, target_meta_label = next(target_meta_iter)

                target_meta_link, target_meta_label = (
                    target_meta_link.to(device),
                    target_meta_label.to(device),
                )
                optimizer.zero_grad()
                target_out = model(
                    source_edge_index,
                    target_train_edge_index,
                    target_meta_link,
                    is_source=False,
                ).squeeze()
                meta_loss = criterion(target_out, target_meta_label).mean()

                for (source_link, source_label), (target_link, target_label) in zip(
                    source_meta_loader, target_meta_loader
                ):
                    source_link, source_label = source_link.to(device), source_label.to(
                        device
                    )
                    target_link, target_label = target_link.to(device), target_label.to(
                        device
                    )
                    weight_source = perceptor(source_link[1], source_edge_index, model)

                    optimizer.zero_grad()
                    source_out = model(
                        source_edge_index,
                        target_train_edge_index,
                        source_link,
                        is_source=True,
                    ).squeeze()
                    target_out = model(
                        source_edge_index,
                        target_train_edge_index,
                        target_link,
                        is_source=False,
                    ).squeeze()
                    source_loss = (
                        criterion(source_out, source_label).reshape(-1, 1)
                        * weight_source
                    ).sum()
                    target_loss = criterion(target_out, target_label).mean()
                    meta_train_loss = (
                        source_loss + target_loss if args.use_meta else target_loss
                    )
                    break

                torch.cuda.empty_cache()
                meta_optimizer.step(
                    train_loss=meta_train_loss,
                    val_loss=meta_loss,
                    aux_params=list(perceptor.parameters()),
                    parameters=model_param,
                    return_grads=True,
                    entropy=None,
                )
        train_auc = evaluate(
            "Train",
            model,
            source_edge_index,
            target_train_edge_index,
            target_train_link,
            target_train_label,
        )
        val_auc = evaluate(
            "Valid",
            model,
            source_edge_index,
            target_train_edge_index,
            target_valid_link,
            target_valid_label,
        )

        logging.info(
            f"[Epoch: {epoch}]Train Loss: {loss:.4f}, Train AUC: {train_auc:.4f}, Valid AUC: {val_auc:.4f}"
        )
        wandb.log(
            {
                "loss": loss,
                "train_auc": train_auc,
                "val_auc": val_auc
            },
            step=epoch,
        )

        early_stopping(val_auc, model)
        if early_stopping.early_stop:
            logging.info("Early stopping")
            break

        lr_scheduler.step()

    model = load_model(args).to(device)
    test_auc = evaluate(
        "Test",
        model,
        source_edge_index,
        target_train_edge_index,
        target_test_link,
        target_test_label,
    )
    logging.info(f"Test AUC: {test_auc:.4f}")
    wandb.log({"Test AUC": test_auc})

    '''
    # === 額外評估熱門 / 冷門商品上的 AUC ===
    test_popular_idx = data.split_mask["test_popular"].cpu().nonzero(as_tuple=True)[0]
    test_unpopular_idx = data.split_mask["test_unpopular"].cpu().nonzero(as_tuple=True)[0]

    test_popular_link = data.target_link[:, test_popular_idx]
    test_popular_label = data.target_label[test_popular_idx]

    test_unpopular_link = data.target_link[:, test_unpopular_idx]
    test_unpopular_label = data.target_label[test_unpopular_idx]

    test_popular_auc = evaluate(
        "Test (Popular)",
        model,
        source_edge_index,
        target_train_edge_index,
        test_popular_link,
        test_popular_label,
    )
    test_unpopular_auc = evaluate(
        "Test (Unpopular)",
        model,
        source_edge_index,
        target_train_edge_index,
        test_unpopular_link,
        test_unpopular_label,
    )

    wandb.log({
        "Test Popular AUC": test_popular_auc,
        "Test Unpopular AUC": test_unpopular_auc,
    })

    test_hr_10 = evaluate_hr_at_k(model, source_edge_index, target_train_edge_index, data, k=10)
    test_hr_15 = evaluate_hr_at_k(model, source_edge_index, target_train_edge_index, data, k=15)
    test_hr_20 = evaluate_hr_at_k(model, source_edge_index, target_train_edge_index, data, k=20)
    test_hr_25 = evaluate_hr_at_k(model, source_edge_index, target_train_edge_index, data, k=25)

    logging.info(f"Test HR@10: {test_hr_10:.4f}")
    logging.info(f"Test HR@15: {test_hr_15:.4f}")
    logging.info(f"Test HR@20: {test_hr_20:.4f}")
    logging.info(f"Test HR@25: {test_hr_25:.4f}")

    wandb.log({
        "Test HR@10": test_hr_10,
        "Test HR@15": test_hr_15,
        "Test HR@20": test_hr_20,
        "Test HR@25": test_hr_25,
    })

    

    for k in [10, 15, 20, 25]:
        hr_popular = evaluate_hr_at_k(model, source_edge_index, target_train_edge_index, test_popular_data, k=k)
        hr_unpopular = evaluate_hr_at_k(model, source_edge_index, target_train_edge_index, test_unpopular_data, k=k)

        logging.info(f"Test HR@{k} (Popular): {hr_popular:.4f}")
        logging.info(f"Test HR@{k} (Unpopular): {hr_unpopular:.4f}")

        wandb.log({
            f"Test HR@{k} (Popular)": hr_popular,
            f"Test HR@{k} (Unpopular)": hr_unpopular,
        })
    '''
    # === 額外評估熱門 / 冷門商品上的 HR@10 ===
    test_popular_data = data.clone()
    test_unpopular_data = data.clone()

    test_popular_data.split_mask["test"] = data.split_mask["test_popular"]
    test_unpopular_data.split_mask["test"] = data.split_mask["test_unpopular"]

    for k in [10, 15, 20, 25]:
        ndcg_all = evaluate_ndcg_at_k(model, source_edge_index, target_train_edge_index, data, k=k)
        wandb.log({f"Test NDCG@{k}": ndcg_all})

        ndcg_popular = evaluate_ndcg_at_k(model, source_edge_index, target_train_edge_index, test_popular_data, k=k)
        ndcg_unpopular = evaluate_ndcg_at_k(model, source_edge_index, target_train_edge_index, test_unpopular_data, k=k)

        logging.info(f"Test NDCG@{k} (Popular): {ndcg_popular:.4f}")
        logging.info(f"Test NDCG@{k} (Unpopular): {ndcg_unpopular:.4f}")

        wandb.log({
            f"Test NDCG@{k} (Popular)": ndcg_popular,
            f"Test NDCG@{k} (Unpopular)": ndcg_unpopular,
        })
