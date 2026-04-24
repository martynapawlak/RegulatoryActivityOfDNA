import pandas as pd 
from sklearn.model_selection import train_test_split

# split 70/15/15 - train, test, val

df = pd.read_csv("dataset.tsv", sep="\t")

train_val_df, test_df = train_test_split(df, test_size = 0.15, random_state = 42, stratify=df['is_active'])

train_df, val_df = train_test_split(train_val_df, test_size=0.176, random_state = 42, stratify=train_val_df['is_active'])

train_df.to_csv("train.tsv", sep ="\t", index=False)
val_df.to_csv("val.tsv", sep ="\t", index=False)
test_df.to_csv("test.tsv", sep ="\t", index=False)