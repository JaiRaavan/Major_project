import pandas as pd
import numpy as np

data = pd.read_csv("./results/test_semevalA_520.csv")
temp = 0

print(data.head())
print(data.tail())
for index, row in data.iterrows():
    print(row)
    break

nan_rows = data[data['Final Judgement'].isna()]

if not nan_rows.empty:
    first_index = nan_rows.index[0]
    first_row = nan_rows.iloc[0]
    
    print(f"Index: {first_index}")
    print("Row Data: ")
    print(first_row)
else:
    print("No Nan value found")