import numpy as numpy
import csv
import sys
import data_ingest


intracranial_hemorrhage_subtypes = [
    "epidural_hemorrhage",
    "intraparenchymal_hemorrhage",
    "intraventricular_hemorrhage",
    "subarachnoid_hemorrhage",
    "subdural_hemorrhage",
    "any"
]


def main():
    with open('../submissions/dry_run.csv', 'a+', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Id','Label'])

    for filename in tqdm(os.listdir(data_ingest.s1_test_path)): 
        standardized_array = data_ingest.read_dicom(filename)

        for subtype in intracranial_hemorrhage_subtypes:
            readable_id = [filename[:-4] + "_" + subtype
            writer.writerow(readable_id, 0])

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