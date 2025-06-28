import pandas as pd

# Load fake and real news
fake_df = pd.read_csv("data/Fake.csv")
real_df = pd.read_csv("data/True.csv")

# Add labels
fake_df['label'] = 'Fake'
real_df['label'] = 'Real'

# Optionally drop unused columns
fake_df = fake_df[['text', 'label']]
real_df = real_df[['text', 'label']]

# Combine the datasets
combined_df = pd.concat([fake_df, real_df], ignore_index=True)

# Save to a single file
combined_df.to_csv("data/news.csv", index=False)

print("✅ Combined dataset created as data/news.csv")
