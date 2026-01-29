import os
import random
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from DataLoad import RNADataProcessor, SingleTissueDataset, MultiTissueDataset
from model import HAC_Net


warnings.filterwarnings('ignore')
BASE_DIR = '/home'
DATA_FILE = './mouse.xlsx'
MODEL_SAVE_PATH = './best_model_mouse.pth'
RESULT_SAVE_PATH = './test_results_mouse.csv'


MIN_EPOCHS = 50
MAX_EPOCHS = 200
PATIENCE = 15
LR_PATIENCE = 5
LR_FACTOR = 0.5


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)



def compute_metrics(preds, targets):
    preds, targets = np.array(preds), np.array(targets)
    mse = mean_squared_error(targets, preds)
    r2 = r2_score(targets, preds)
    if len(preds) > 1 and np.std(preds) > 1e-9 and np.std(targets) > 1e-9:
        pcc, _ = pearsonr(targets, preds)
        scc, _ = spearmanr(targets, preds)
    else:
        pcc, scc = 0.0, 0.0
    return mse, r2, pcc, scc


def adapt_and_test(model, tissue_ds, device, support_ratio=0.1, adapt_steps=50, lr=0.01):
    total = len(tissue_ds)
    n_support = int(total * support_ratio)
    n_query = total - n_support
    if n_support < 5: return None
    support_ds, query_ds = random_split(tissue_ds, [n_support, n_query])
    support_loader = DataLoader(support_ds, batch_size=16, shuffle=True)
    query_loader = DataLoader(query_ds, batch_size=32, shuffle=False)
    with torch.no_grad():
        init_emb = model.tissue_embed.weight.mean(dim=0).detach().clone()
    ctx_emb = nn.Parameter(init_emb.unsqueeze(0))
    optimizer = torch.optim.Adam([ctx_emb], lr=lr)
    criterion = nn.MSELoss()
    model.eval()
    for _ in range(adapt_steps):
        for batch in support_loader:
            u5, cds, u3 = batch['u5'].to(device), batch['cds'].to(device), batch['u3'].to(device)
            target = batch['te'].to(device)
            batch_ctx = ctx_emb.expand(u5.size(0), -1)
            pred = model(u5, cds, u3, external_ctx_emb=batch_ctx)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    preds, targets = [], []
    with torch.no_grad():
        for batch in query_loader:
            u5, cds, u3 = batch['u5'].to(device), batch['cds'].to(device), batch['u3'].to(device)
            target = batch['te'].to(device)
            batch_ctx = ctx_emb.expand(u5.size(0), -1)
            pred = model(u5, cds, u3, external_ctx_emb=batch_ctx)
            preds.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
    return compute_metrics(preds, targets)


def main():
    BATCH_SIZE = 64
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Running on device: {DEVICE}")

    if not os.path.exists(DATA_FILE):
        print(f"[Error] File not found: {DATA_FILE}")
        return

    processor = RNADataProcessor(DATA_FILE)
    all_tissues = processor.te_cols
    random.shuffle(all_tissues)
    n = len(all_tissues)
    n_val, n_test = int(n * 0.1), int(n * 0.1)
    n_train = n - n_val - n_test
    train_tissues = all_tissues[:n_train]
    val_tissues = all_tissues[n_train:n_train + n_val]
    test_tissues = all_tissues[n_train + n_val:]

    train_datasets = []
    train_map = {t: i for i, t in enumerate(train_tissues)}
    for t in train_tissues:
        df_t = processor.get_tissue_data(t)
        train_datasets.append(SingleTissueDataset(df_t, t, train_map))

    print("[Info] Preparing Training Data...")
    train_loader = DataLoader(MultiTissueDataset(train_datasets), batch_size=BATCH_SIZE, shuffle=True)

    model = HAC_Net(num_train_tissues=len(train_tissues)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=LR_FACTOR, patience=LR_PATIENCE
    )

    best_val_pcc = -999
    patience_counter = 0

    print(f"\n[Start Training] Min Epochs: {MIN_EPOCHS}, Max Epochs: {MAX_EPOCHS}, Early Stop Patience: {PATIENCE}")

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{MAX_EPOCHS} [Train]", unit="batch")

        for batch in pbar:
            u5, cds, u3 = batch['u5'].to(DEVICE), batch['cds'].to(DEVICE), batch['u3'].to(DEVICE)
            tid, target = batch['tissue_id'].to(DEVICE), batch['te'].to(DEVICE)
            optimizer.zero_grad()
            pred = model(u5, cds, u3, tissue_id=tid)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)

        val_pccs = []
        val_r2s = []

        for t in tqdm(val_tissues, desc=f"Epoch {epoch + 1}/{MAX_EPOCHS} [Val  ]", leave=False):
            df_t = processor.get_tissue_data(t)
            if len(df_t) < 50: continue
            ds = SingleTissueDataset(df_t, t, {})
            res = adapt_and_test(model, ds, DEVICE, support_ratio=0.1)
            if res:
                val_pccs.append(res[2])
                val_r2s.append(res[1])

        avg_pcc = np.mean(val_pccs) if val_pccs else 0
        avg_r2 = np.mean(val_r2s) if val_r2s else 0

        scheduler.step(avg_pcc)
        current_lr = optimizer.param_groups[0]['lr']

        tqdm.write(
            f"Epoch {epoch + 1} Summary: Loss={avg_loss:.4f}, Val PCC={avg_pcc:.4f}, Val R2={avg_r2:.4f}, LR={current_lr:.6f}")

        if avg_pcc > best_val_pcc:
            best_val_pcc = avg_pcc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            tqdm.write(f"  >>> Best Model Saved (PCC: {best_val_pcc:.4f})")
            if epoch >= MIN_EPOCHS:
                patience_counter = 0
        else:
            if epoch >= MIN_EPOCHS:
                patience_counter += 1
                tqdm.write(f"  [Info] Early Stopping Counter: {patience_counter}/{PATIENCE}")
                if patience_counter >= PATIENCE:
                    print(f"\n[Early Stopping] Triggered at Epoch {epoch + 1}")
                    break

    print("\n[Final Testing]")
    if os.path.exists(MODEL_SAVE_PATH): model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    results = []
    for t in tqdm(test_tissues, desc="Testing"):
        df_t = processor.get_tissue_data(t)
        if len(df_t) < 50: continue
        ds = SingleTissueDataset(df_t, t, {})
        res = adapt_and_test(model, ds, DEVICE, support_ratio=0.1)
        if res: results.append({'Tissue': t, 'MSE': res[0], 'R2': res[1], 'PCC': res[2], 'SCC': res[3]})

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(RESULT_SAVE_PATH, index=False)
        print(f"\n[Success] Saved to {RESULT_SAVE_PATH}")
        print(f"Overall Mean PCC: {res_df['PCC'].mean():.4f}")
        print(f"Overall Mean R2 : {res_df['R2'].mean():.4f}")


if __name__ == '__main__':
    main()