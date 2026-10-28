import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("AI Moisture Detection/moistnetdata.csv")

train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42) 
# means 30% of the rows go into temp_df, and 70% into train_df
test_df, val_df = train_test_split(temp_df, test_size=0.50, random_state=42) 
# 50% of the remaining rows go into val_df, and 50% into test_df

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)
# convert to csv's

print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")
print("Splitting completed successfully.")
# print the number of samples in each set