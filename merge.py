import os
import shutil

base_path = "Data"
splits = ["Train", "Test", "Validation"]

for split in splits:
    split_path = os.path.join(base_path, split)

    for folder in os.listdir(split_path):
        folder_path = os.path.join(split_path, folder)

        if os.path.isdir(folder_path):
            class_name = folder.split("_")[0]
            new_class_path = os.path.join(split_path, class_name)

            if folder == class_name:
                continue  # skip already merged folder

            os.makedirs(new_class_path, exist_ok=True)

            for file in os.listdir(folder_path):
                shutil.move(
                    os.path.join(folder_path, file),
                    os.path.join(new_class_path, file)
                )

            os.rmdir(folder_path)

print("Merge complete.")
