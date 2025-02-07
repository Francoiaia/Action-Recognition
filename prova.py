# -*- coding: utf-8 -*-
import os
import csv


def create_dataset_csv(root_dir, output_file):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])  # Header

        # Walk through all directories
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if any(filename.endswith(".mp4") for filename in filenames):
                # Determine the label based on the directory name
                # if 'sit' in dirpath.lower():
                #     label = 'sit'
                if "fall" in dirpath.lower() and not "no_fall" in dirpath.lower():
                    label = "fall"
                else:
                    label = "nofall"
                #
                # elif 'lay' in dirpath.lower():
                #     label = 'lay'
                # else:
                #     continue  # Skip if not in a relevant directory

                # Write all MP4 files in this directory
                for filename in filenames:
                    if filename.endswith(".mp4"):
                        full_path = os.path.join(dirpath, filename)
                        # Get relative path from root directory
                        rel_path = os.path.relpath(full_path, root_dir)
                        writer.writerow([rel_path, label])


# Use the function
create_dataset_csv("C:/work/repo/Datasets/magic_fall_dataset", "src/video/magic_fall_dataset_2class_train.csv")
