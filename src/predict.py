import numpy as numpy
import csv
import sys


test_data_path = "../data/stage_1_test_images/"
intracranial_hemorrhage_subtypes = [
    "epidural_hemorrhage",
    "intraparenchymal_hemorrhage",
    "intraventricular_hemorrhage",
    "subarachnoid_hemorrhage",
    "subdural_hemorrhage",
    "any"
]


def read_dicom(filename):
    """
    Transform a medical DICOM file to a standardized pixel based array
    """
    img = np.array(pydicom.dcmread(test_data_path + filename).pixel_array, dtype=float).T
    mean = img.mean()
    std = img.std()
    #perfmorming standardization by subtracting the mean and dividing the s.d.
    standardized_array = np.divide(np.subtract(img,mean),std)
    return standardized_array


def main():
    with open('../submissions/dry_run.csv', 'a+', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Id','Label'])

    for filename in tqdm(os.listdir(test_data_path)): 
        # read_dicom(filename) comment out for dry run

        for subtype in intracranial_hemorrhage_subtypes:
            readable_id = [filename[:-4] + "_" + subtype
            writer.writerow(readable_id, 0])
    sys.exit()

if __name__ == "__main__":
    main()
        


        # expanded_array = standardized_array[np.newaxis, ..., np.newaxis]
        # msk = model.predict(expanded_array)
        # msk = np.squeeze(np.round(msk)) #remove axis w/ dims of 1, round mask for given probbabilties to [0,1]
        
        # if not np.any(msk) == True:
        #     writer.writerow([filename[:-4], '-1']) #case for no pnuemothorax found
        #     print('none found...')
        # else:
        #     rle = mask2rle(msk, 1024, 1024)
        #     writer.writerow([filename[:-4], rle])