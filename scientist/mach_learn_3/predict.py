import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# import csv and sort on date
df = pd.read_csv('sphist.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

day5_avg = []
day5_std = []
for i, row in df.iterrows():
    if i < 5:
        day5_avg.append(0)
        day5_std.append(0)
    else:
        avg = df.iloc[i-5:i]['Close'].mean()
        std = df.iloc[i-5:i]['Close'].std()
        day5_avg.append(avg)
        day5_std.append(std)
df['5_day_avg'] = day5_avg
df['5_day_std'] = day5_std

day30_avg = []
for i, row in df.iterrows():
    if i < 30:
        day30_avg.append(0)
    else:
        avg = df.iloc[i-30:i]['Close'].mean()
        day30_avg.append(avg)
df['30_day_avg'] = day30_avg

day365_avg = []
day365_std = []
for i, row in df.iterrows():
    if i < 365:
        day365_avg.append(0)
        day365_std.append(0)
    else:
        avg = df.iloc[i-365:i]['Close'].mean()
        std = df.iloc[i-365:i]['Close'].std()
        day365_avg.append(avg)
        day365_std.append(std)
df['365_day_avg'] = day365_avg
df['365_day_std'] = day365_std

df['5_365_avg_ratio'] = df['5_day_avg'] / df['365_day_avg']
df['5_365_std_ratio'] = df['5_day_std'] / df['365_day_std']

df = df[df['Date'] > datetime(year=1951, month=1, day=3)]
df.dropna(axis=0)
train = df[df['Date'] < datetime(year=2013, month=1, day=1)]
test = df[df['Date'] >= datetime(year=2013, month=1, day=1)]

features = ['5_day_avg', '30_day_avg', '365_day_avg', '5_365_avg_ratio', '5_day_std', '365_day_std', '5_365_std_ratio']
target = 'Close'

lr = LinearRegression()
lr.fit(train[features], train[target])
predictions = lr.predict(test[features])

mae = mean_absolute_error(test[target], predictions)
print(mae)
