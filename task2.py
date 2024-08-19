import pandas as pd
threshold_value=8.7
data = pd.read_csv("D:\\vasanth\\pythons\\NM\\NLU_dataset\\College.csv")
filtered_df = data[data['Public Rating'] > threshold_value]
df_filled = filtered_df.fillna(0)
summary_stats = df_filled.describe()

print(filtered_df.head())
print(summary_stats)