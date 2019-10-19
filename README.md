# intracranial-hemorrhage-detection
Repo for RSNA intracranial hemorrhage detection. Instructions for reproducability can be found below.

## Overview
Intracranial hemorrhage (bleeding within the cranium) accounts for ~10% of strokes in the U.S., where stroke is the fifth-leading cause of death. There are several types of hemorrhage and indentifying/locating them in a patient quickly is not only a matter of life and death, but speed also plays a critical part in the quality of life a survior can expext post recovery.

Diagnosing and locating an intracranial hemorrhage from neurological symptoms (e.g. severe headache or loss of consciousness) and medical imagery is a complicated, time consuming process requiring a highly trained spcialist. A technology solution would enhance the speed and diagnostic ability of medical practioners as well as potentially provide diagnostic ability or relief to patients who are not near an expert. Our goal is to build a model and system which detecs acute intracranial hemorrhages and its subtypes. 
![hemorrhage_subtypes](./images/hemorrhage-types.png "Hemorrhage Subtypes")

## Data
The dataset has been provided by the Radiological Society of North America (RSNA®) in collaboration with members of the American Society of Neuroradiology and MD.ai.

## Data Augmentation
Data is augmented via random flips (vertical and horizontal), rotations (+/- 10 degrees), Salt&Pepper noise, and affine transformations (deformation) to help capture the invariants. This also helps to upsample class 1 data. We also window channels sepretly. In the image below we show a 3 channel img combined via np.dstack(), and the corrosponding individual channels. The windowing used on this particular hounsfield normalized data are: brain, subdural, and blood. We also apply image normalization to help the CNN. 

![sample_data_augmentation](./images/sample_data_augmentation.png "Sample Data Augmentation")

Furthermore we also downsample class 0 data by randomly selecting images to match the number of class 1 training data. When training binary cross entropy for "any" class 1 subtype, we set all subtypes to class 1 and an equal amount of class 0 to 0 

## Model
* Test Submission: First approach was the test the submission script and manually mark each subtype, including All, with a probability value of 0.15
* Naive Baseline: Pass


## Evaluation
Model evaluated using a weighted multi-label logarithmic loss (same as cross-entropy for all intents and purposes). Using the minmax rule to avoid undefinned predictions at {0,1}, offset by a small epsilon: max(min(p, 1−10^−15), 10^-15).

## Team
- Chris Chen
- Tony Reina
- Kyle Shannon

## Getting Started
Instructions for deploying our codebase and reproducing our results:
1. Run the Docker script to create a GPU ready linux container. (We assume you will be using Nvidia GPUs on a Linux based system.
2. Ensure that the conda environment was set up properly and matches the .yaml environment configuration. 
3. Set your paths to the training and test data in the src/config.ini file. 
4. You can recreate our exact train/validate/test splits using the ```create_label_file.py``` script, alternatively we have provided those csv files for convienance.

## Model Training Instructions:
start training via:
python train.py {model-name}-{dims}-{loss}-{subtype}-{monthDay-version}
e.g.
`$ python train.py resnet50-dim224x224-bce-intraparenchymal-oct18v1`

After training is complete, please create a folder in the google drive shared folder with the models name (e.g. resnet50-dim224x224-bce-intraparenchymal-oct18v1) in that folder please upload the model.pb, model's tensorboard folder, and the current src/ folder that was used to train the model, that way we are ensured to know which copy of the data_loader etc was used to train the model. Note you might need to `$ sudo chown user ./dir/*` the models and tensboard folder, because the permissions may be different from the docker container when writing to local disk.

In the google spreadsheet, you can add more info about the model, anything interesting you noted and be on your merry way. Thanks!
I used the following arguments for IH subtypes on ResNet50:
```
BATCH_SIZE = 32
EPOCHS = 15 
DIMS = (224,224)
training_data_gen = DataGenerator(csv_filename=TRAIN_CSV,
                                    data_path=DATA_DIRECTORY,
                                    batch_size=BATCH_SIZE,
                                    dims=DIMS,
                                    augment=True,
                                    balance_data = True,
                                    subtype = "intraparenchymal",  ####### <-- change this
                                    channel_types = ['subdural','soft_tissue','brain']) ### <-- we are using these windows?
validation_data_gen = DataGenerator(csv_filename=VALIDATE_CSV,
                                    data_path=DATA_DIRECTORY,
                                    batch_size=BATCH_SIZE,
                                    dims=DIMS,
                                    augment=False,
                                    balance_data = True,
                                    subtype = "intraparenchymal",  ####### <-- change this
                                    channel_types = ['subdural','soft_tissue','brain']) ### <-- we are using these windows?
```
