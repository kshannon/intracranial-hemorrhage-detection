#!/usr/bin/env python
# Custom module for dealing with global project paths and functions related to injesting and accessing raw data

import os
import sys
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
tqdm.pandas()


CSV_DIRECTORY = "../../data"
DATA_DIRECTORY = "../../data/stage_1_train_images/"
DUPLICATES = ['ID_a64d5deed.dcm','ID_921490062.dcm','ID_489ae4179.dcm','ID_854fba667.dcm','ID_854fba667.dcm']
MASTER_CSV = os.path.join(CSV_DIRECTORY, "master_training.csv")

if os.path.exists(MASTER_CSV):
    master_df = pd.read_csv(MASTER_CSV)
else:
    train_df = pd.read_csv(os.path.join(CSV_DIRECTORY, "stage_1_train.csv"))
    train_df["filename"] = train_df["ID"].apply(lambda st: "ID_" + st.split("_")[1] + ".dcm")
    train_df["type"] = train_df["ID"].apply(lambda st: st.split("_")[2])

    # New pandas dataframe with the target labels organized into a numpy array
    master_df = train_df[["Label", "filename", "type"]]\
                    .drop_duplicates()\
                    .pivot(index="filename", columns="type", values="Label")\
                    .reset_index()

    master_df["targets"] = master_df.progress_apply(lambda x: np.array([float(x["epidural"]),
                                                                    float(x["intraparenchymal"]),
                                                                    float(x["intraventricular"]),
                                                                    float(x["subarachnoid"]),
                                                                    float(x["subdural"]),
                                                                    float(x["any"])]), axis=1)

    master_df["any"] = master_df.progress_apply(lambda x: float(x["any"]), axis=1)

    # save the master df as a csv
    master_df.to_csv(os.path.join(CSV_DIRECTORY, "master_training.csv"), index=False)
    print("Created and saved a master training CSV to disk. You're welcome...")

# We have a master DF, lets create two sub DFs for 
class1_df = master_df.loc[master_df['any'] == 1] # 97103 class 1 (14% of the data)
class0_df = master_df.loc[master_df['any'] == 0] # 577155 class 0
assert class0_df.shape[0] + class1_df.shape[0] == master_df.shape[0]

# Shuffle and randomly undersample class 0
class0_df = class0_df.sample(frac=1, random_state=13).reset_index(drop=True)
class0_df = class0_df.sample(n=class1_df.shape[0], random_state=13) #50/50 split

# Reconstitute balanced dataset, shuffle whole dataset
balanced_df = pd.concat([class1_df, class0_df], ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=13).reset_index(drop=True)

# Create random train/validation sets and save to csv
train_df = balanced_df.sample(frac=0.85, random_state=13) #random state is a seed value
validation_df = balanced_df.drop(train_df.index)
test_df = None #TODO if we require or want this,possibly for CV, but for now train/val are fine.
assert train_df.shape[0] + validation_df.shape[0] == balanced_df.shape[0]

train_df[["filename", "targets", "any"]].to_csv("../src/training.csv", index=False, header=False)
validation_df[["filename", "targets", "any"]].to_csv("../src/validation.csv", index=False, header=False)
# test_df[["filename", "targets", "any"]].to_csv("../src/validation.csv", index=False)
