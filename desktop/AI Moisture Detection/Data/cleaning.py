import pandas as pd
import os
os.chdir(os.path.dirname(__file__))  # sets working dir to this script’s folder

def clean_csv(input_csv, output_csv):
    # Load CSV
    df = pd.read_csv(input_csv)

    # Keep only the needed columns
    df = df[['folder', 'image']]

    # Fix Windows paths (\ -> /)
    df['image'] = df['image'].str.replace('\\', '/', regex=False)

    # Save back
    df.to_csv(output_csv, index=False)
    print(f"✅ Cleaned and saved: {output_csv}")

# Run for all your splits
clean_csv("train.csv", "train_clean.csv")
clean_csv("val.csv", "val_clean.csv")
clean_csv("test.csv", "test_clean.csv")