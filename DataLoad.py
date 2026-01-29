import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class RNADataProcessor:
    def __init__(self, file_path, vocab={'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}, max_len=4000):
        print(f"[Info] Loading data from {file_path}...")
        if file_path.endswith('.xlsx'):
            self.raw_df = pd.read_excel(file_path)
        else:
            self.raw_df = pd.read_csv(file_path)

        self.vocab = vocab
        self.max_len = max_len
        self.te_cols = [c for c in self.raw_df.columns if c.startswith('TE_')]
        self.all_tissues = self.te_cols
        print(f"[Info] Found {len(self.te_cols)} tissues.")

        tqdm.pandas(desc="Processing Sequences")
        self.raw_df['seq_data'] = self.raw_df.progress_apply(self._process_sequence, axis=1)

    def _process_sequence(self, row):
        full_seq = str(row['tx_sequence']).upper()
        u5_sz = int(row['utr5_size'])
        cds_sz = int(row['cds_size'])
        u5_seq = full_seq[:u5_sz][-500:]
        cds_seq = full_seq[u5_sz: u5_sz + cds_sz][:2000]
        u3_seq = full_seq[u5_sz + cds_sz:][:1500]

        def to_indices(seq): return [self.vocab.get(b, 4) for b in seq]

        return {'u5': to_indices(u5_seq), 'cds': to_indices(cds_seq), 'u3': to_indices(u3_seq)}

    def get_tissue_data(self, tissue_name):
        return self.raw_df[['seq_data', tissue_name]].dropna()


class SingleTissueDataset(Dataset):
    def __init__(self, data_df, tissue_name, tissue_id_map):
        self.data = data_df
        self.tissue_name = tissue_name
        self.tissue_idx = tissue_id_map.get(tissue_name, 0)

    def __len__(self): return len(self.data)

    def pad_seq(self, seq, target_len):
        if len(seq) > target_len: return seq[:target_len]
        return seq + [4] * (target_len - len(seq))

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        seq_dict = row['seq_data']
        te_val = row[self.tissue_name]
        u5 = torch.tensor(self.pad_seq(seq_dict['u5'], 500), dtype=torch.long)
        cds = torch.tensor(self.pad_seq(seq_dict['cds'], 2000), dtype=torch.long)
        u3 = torch.tensor(self.pad_seq(seq_dict['u3'], 1500), dtype=torch.long)
        return {'u5': u5, 'cds': cds, 'u3': u3, 'tissue_id': torch.tensor(self.tissue_idx, dtype=torch.long),
                'te': torch.tensor(te_val, dtype=torch.float32)}


class MultiTissueDataset(Dataset):
    def __init__(self, datasets):
        self.samples = []
        for ds in datasets:
            for i in range(len(ds)): self.samples.append((ds, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ds, sample_idx = self.samples[idx]
        return ds[sample_idx]