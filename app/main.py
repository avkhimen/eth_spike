import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import warnings
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.simplefilter('ignore')

df = pd.read_csv('data/original_files/ETHXBT_60.csv', header=None,
                 names=['unix_timestamp','open_price','high_price','low_price','close_price','other_1','other_2'])

df = df.dropna()

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

df['close_price_perc'] = df['close_price'].pct_change()

df = df[1:]

df['close_price_perc'] = df['close_price_perc'] * 100

threshold = 4

df['spike'] = (df['close_price_perc'] > threshold).fillna(False)

df = df[['close_price','spike']]

offset = 4

close_prices_list = sliding_window_view(df['close_price'], offset).tolist()

spike_list = sliding_window_view(df['spike'], 1).tolist()[offset-1:]
spike_list = [elem[0] for elem in spike_list]

X = close_prices_list[:-1]
y_ = spike_list[1:]

y = []
for elem in y_:
    y.append(str(elem))

print(df.head(10))
print(X[:5], len(X))
print(y[:5], len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

clf = svm.SVC()
clf = tree.DecisionTreeClassifier()

clf.fit(X_train, y_train)

yhat = clf.predict(X_test)

acc = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % acc)