import pandas as pd
import numpy as np

df = pd.read_csv('data/original_files/ETHXBT_60.csv', header=None,
                 names=['unix_timestamp','open_price','high_price','low_price','close_price','other_1','other_2'])

df['unix_timestamp'] = df['unix_timestamp'].astype(int)

df['timestamp'] = pd.to_datetime(df['unix_timestamp'], unit='s')

df = df.drop(['unix_timestamp', 'other_1', 'other_2'], axis=1)

df = df[['timestamp','open_price','high_price','low_price','close_price']]

df = df.resample('4H', on='timestamp').agg({
    'open_price': 'first',
    'high_price': 'max',
    'low_price': 'min',
    'close_price': 'last'
    })

df = df[['close_price']]

df = df.pct_change()

df = df[1:]

df['close_price_perc'] = df['close_price'] * 100

threshold = 4

df['spike'] = (df['close_price_perc'] > threshold).fillna(False)

df = df[['close_price','spike']]

print(df)