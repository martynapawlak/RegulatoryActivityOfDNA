import torch
from torch.utils.data import Dataset
import pandas as pd

mapping = {"A":0, "T":1, "C":2, "G":3 }

def encode(seq):
	mapped = []
	for x in seq:
		emb = mapping[x]
		mapped.append(emb)
	return mapped

class DNADataset(Dataset):
	def __init__(self,file):
		self.df = pd.read_csv(file, sep="\t")

		self.sequences = self.df["sequence"].values
		self.y_class = self.df["is_active"].values
		self.y_reg = self.df["rna_dna_ratio"].values

	def __len__(self):
		return len(self.sequences)

	def __getitem(self,idx):
		seq = torch.tensor(encode(self.sequences[idx]), dtype=torch.long)
		y_class = torch.tensor(self.y_class[idx], dtype=torch.float32)
		y_reg = torch.tensor(self.y_reg[idx], dtype=torch.float32)
		return seq, y_class, y_reg



