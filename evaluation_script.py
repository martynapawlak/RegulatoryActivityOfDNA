import torch
import pandas as pd 
import argparse
import sys

from models.model import DNAMultitaskModel
from data.preprocess import one_hot_encode

def evaluate(test_path, model_path):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	checkpoint = torch.load(model_path, map_location=device)

	model = DNAMultitaskModel()
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.to(device)
	model.eval()

	train_mean = checkpoint['train_mean']
	train_std = checkpoint['train_std']

	try:
		df = pd.read_csv(test_path, sep='\t')
	except Exception as e:
		print(f"error while opening the file: {e}", file=sys.stderr)
		return
	print("id\tpredicted_is_active\tpredicted_rna_dna_ratio")
	with torch.no_grad():
		for index, row in df.iterrows():
			seq_id = row['seq_id']
			seq_str = row['sequence']
			seq_tensor = one_hot_encode(seq_str).unsqueeze(0).to(device)
			out_class, out_reg = model(seq_tensor)
			class_prob = out_class.squeeze().item()
			predicted_is_active = 1 if class_prob > 0.5 else 0
			normalized_ratio = out_reg.squeeze().item()
			predicted_rna_dna_ratio = (normalized_ratio * train_std) + train_mean
			print(f"{seq_id}\t{predicted_is_active}\t{predicted_rna_dna_ratio}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate cnn model")
    parser.add_argument('test_path', type=str, help="path to test file")
    parser.add_argument('model_path', type=str, help="path to model")
    
    args = parser.parse_args()
    
    evaluate(args.test_path, args.model_path)

