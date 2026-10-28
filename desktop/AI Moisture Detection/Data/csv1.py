import os
import csv

root_dir = "C:/Users/Arham/Downloads/WoodChip-MC" 

output_csv = "moistnetdata.csv"

if not os.path.exists(root_dir):
    print(f"Root directory does not exist: {root_dir}")
    exit(1)

rows=[]

# Process Source 1
source1_path = os.path.join(root_dir, "source_1")
for folder in os.listdir(source1_path):
    if not folder.isdigit():
        continue
    label = float(folder)
    folder_path = os.path.join(source1_path, folder)
    for container in os.listdir(folder_path):
        container_num = container.split("_")[1]
        container_path = os.path.join(folder_path, container)
        for image in os.listdir(container_path):
            image_path = os.path.join(container_path, image)
            rows.append(["source_1", "-", label, container_num, image_path])
            
# Process Source 2
source2_dir = os.path.join(root_dir, "source_2")
for batch in os.listdir(source2_dir):
    batch_path = os.path.join(source2_dir, batch)
    # Skip anything that’s not a batch folder (e.g. hidden files)
    if not os.path.isdir(batch_path):
        continue

    # Example batch folder name: "batch_one" or "batch_two"
    batch_label = batch.split("_")[1]  
    for folder in os.listdir(batch_path):
        folder_path = os.path.join(batch_path, folder)
        
        if not folder.isdigit():
            continue

        label = float(folder)   # the actual moisture
        for container in os.listdir(folder_path):
            container_path = os.path.join(folder_path, container)
            if not os.path.isdir(container_path):
                continue
            container_num = container.split("_")[1]  # "container_2" → "2"

            # Now loop the images
            for image in os.listdir(container_path):
                image_path = os.path.join(container_path, image)
                rows.append([
                    "source_2",
                    batch_label,
                    label,
                    container_num,
                    image_path
                ])

with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["source", "batch", "folder", "container", "image"])
    writer.writerows(rows)

print(f"Data has been written to {output_csv} with {len(rows)} entries successfully!")
