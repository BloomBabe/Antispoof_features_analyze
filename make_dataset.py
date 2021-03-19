import os
from shutil import copyfile
from tqdm import tqdm

main_dataset_dir = 'D:\\LCC_FASD\\LCC_FASD\\LCC_FASD_training'
new_dataset_dir = 'D:\\JetBrains_PyCharm\\ml_university\\ml_antispoof\\data'

for label in os.listdir(main_dataset_dir):
    LENGTH_LABEL = 100
    new_dataset_file = []
    label_dir = os.path.join(main_dataset_dir, label)

    for filename in tqdm(os.listdir(label_dir)):
        if len(new_dataset_file) == 0:
            new_dataset_file.append(os.path.join(label_dir, filename))
            continue

        SAME = False
        for old_filename in new_dataset_file:
            if filename.split('_')[-3] == old_filename.split('_')[-3]:
                SAME = True
        if SAME:
            continue
        else:
            new_dataset_file.append(os.path.join(label_dir, filename))

    new_label_dir = os.path.join(new_dataset_dir, label)

    if not os.path.isdir(new_label_dir):
        os.makedirs(new_label_dir)

    for filepath in new_dataset_file[:LENGTH_LABEL]:
        new_file_path = os.path.join(new_label_dir, os.path.basename(filepath))
        copyfile(filepath, new_file_path)
