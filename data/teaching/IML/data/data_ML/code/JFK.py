import pandas as pd
from dateutil import parser
import matplotlib.pyplot as plt

cols = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance']

dfJ = pd.read_csv('yellow_tripdata_2017-01.csv', usecols=cols)
dfF = pd.read_csv('yellow_tripdata_2017-02.csv', usecols=cols)
dfM = pd.read_csv('yellow_tripdata_2017-03.csv', usecols=cols)
#dfA = pd.read_csv('yellow_tripdata_2017-04.csv', usecols=cols)
#dfMy = pd.read_csv('yellow_tripdata_2017-05.csv', usecols=cols)

df = dfJ.append(dfF).append(dfM) #.append(dfA) #.append(dfMy)


#236 manhattan upper east side
JFK_MU = df[(df['PULocationID'] == 132) & (df['DOLocationID'] == 236)]

JFK_MU.to_csv("JFKraw.csv", columns=cols)

#JFK_MU = JFK_MU[(JFK_MU['trip_distance'] > 18.3) & (JFK_MU['trip_distance'] < 20.3)]
JFK_MU['weekday'] = JFK_MU['tpep_pickup_datetime'].apply(lambda x : parser.parse(x).weekday())

JFK_MU = JFK_MU[JFK_MU['weekday'] < 5]

pu = [parser.parse(dt) for dt in JFK_MU['tpep_pickup_datetime'].values]
do = [parser.parse(dt) for dt in JFK_MU['tpep_dropoff_datetime'].values]
dur = [(b -a).total_seconds() / 3600.0 for a, b in zip(pu, do)]
startTime = [dt.hour + dt.minute / 60.0 for dt in pu]

plt.scatter(startTime, dur)
plt.show()

#df[(df['PULocationID'] == 132) & (df['DOLocationID'] == 236) & (parser.parse(df['tpep_pickup_datetime']).weekday() == 1)]
