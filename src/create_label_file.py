#!/usr/bin/env python
# Custom module for dealing with global project paths and functions related to injesting and accessing raw data

import os
import pandas as pd
import numpy as np

csv_directory = "../../"
data_directory = "../../stage_1_train_images/"

# Load all of the training files
train_df = pd.read_csv(os.path.join(csv_directory, "stage_1_train.csv"))

train_df["filename"] = train_df["ID"].apply(lambda st: "ID_" + st.split("_")[1] + ".dcm")
train_df["type"] = train_df["ID"].apply(lambda st: st.split("_")[2])

# New pandas dataframe with the target labels organized into a numpy array
pivot_df = train_df[["Label", "filename", "type"]].drop_duplicates().pivot(
    index="filename", columns="type", values="Label").reset_index()

pivot_df["targets"] = pivot_df.apply(lambda x: np.array([float(x["epidural"]),
                                                     float(x["intraparenchymal"]),
                                                     float(x["intraventricular"]),
                                                     float(x["subarachnoid"]),
                                                     float(x["subdural"]),
                                                     float(x["any"])]), axis=1)

print(pivot_df.shape)
print(pivot_df.head(30))

# Do the train, validate, test splits

np.random.seed(816)
all_training_files = os.listdir(data_directory)
# Shuffle files
np.random.shuffle(all_training_files)
number_training_files = len(all_training_files)

# Now we have the actual filenames from the data directory shuffled
# So first 80% are training
# Next 10% are validation
# Last 10% are testing
train_percentage = 0.80
validation_percentage = 0.10
testing_percentage = 1 - train_percentage - validation_percentage

# Get the filenames for each group
train_idx = int(number_training_files*train_percentage)
train_files = all_training_files[0:train_idx]
validate_idx = int(number_training_files*(train_percentage+validation_percentage))
validation_files = all_training_files[train_idx:validate_idx]
test_files = all_training_files[validate_idx:]

# Get the filename/target label pairs for each group
# Note that the "isin" ensures that we have actual files to load
train_df = pivot_df[pivot_df.filename.isin(train_files)]
validation_df = pivot_df[pivot_df.filename.isin(validation_files)]
test_df = pivot_df[pivot_df.filename.isin(test_files)]

# Save the 3 lists to CSV files for use in Keras
train_df[["filename", "targets"]].to_csv("training.csv", index=False)
validation_df[["filename", "targets"]].to_csv("validation.csv", index=False)
test_df[["filename", "targets"]].to_csv("testing.csv", index=False)
