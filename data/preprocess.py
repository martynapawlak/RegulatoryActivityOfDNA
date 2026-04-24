import torch
from torch.utils.data import Dataset
import pandas as pd

mapping = {"A":0, "T":1, "C":2, "G":3}

complement = {"A": "T", "T": "A", "C": "G", "G": "C"}

def reverse_complement(seq):
    return "".join(complement[x] for x in reversed(seq))

def one_hot_encode(seq):
	mapped = []
	for x in seq:
		emb = mapping[x]
		mapped.append(emb)
	tensor = torch.tensor(mapped, dtype=torch.long)
	one_hot = F.one_hot(tensor, num_classes=4).float() # 5 = a+t+g+c+n

	return one_hot.transpose(0,1)


def reverse_complement(seq):
    return "".join(complement[x] for x in reversed(seq))

class DNADataset(Dataset):
	def __init__(self,file, mean_reg=None, std_reg=None, augment=False):
		self.df = pd.read_csv(file, sep="\t")

		self.sequences = self.df["sequence"].values
		self.y_class = self.df["is_active"].values
		self.y_reg = self.df["rna_dna_ratio"].values

		self.augment = augment


		if mean_reg is not None:
			self.y_reg = (self.y_reg - mean_reg) / std_reg

	def __len__(self):
		return len(self.sequences)

	def __getitem__(self,idx):
		seq_str = self.sequences[idx]

		if self.augment and random.random() < 0.5:
				seq_str = reverse_complement(seq_str)

		seq = one_hot_encode(seq_str)

		y_class = torch.tensor(self.y_class[idx], dtype=torch.float32)
		y_reg = torch.tensor(self.y_reg[idx], dtype=torch.float32)
		return seq, y_class, y_reg