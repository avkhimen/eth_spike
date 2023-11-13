import pandas as pd
import numpy as np

df = pd.read_csv('data/original_files/ETHXBT_60.csv', header=None,
                 names=['unix_timestamp','open_price','high_price','low_price','close_price','other_1','other_2'])

print(df)