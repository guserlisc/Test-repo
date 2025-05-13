import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Läs in GTFS-filer (du kan börja med påhittad testdata först)
stops = pd.read_csv('stops.txt')
stop_times = pd.read_csv('stop_times.txt')
trips = pd.read_csv('trips.txt')
calendar = pd.read_csv('calendar.txt')

# Koppla ihop dem
df = stop_times.merge(trips, on='trip_id').merge(calendar, on='service_id').merge(stops, on='stop_id')

# Exempel på hur du kan skapa en dummy-målvariabel (tillfälligt!)
import numpy as np
df['num_crimes'] = np.random.randint(0, 5, size=len(df))  # Ta bort detta när du har brottsdata

# Feature engineering (t.ex. extrahera tid eller plats)
df['hour'] = df['arrival_time'].str[:2].astype(int)
df['weekday'] = df['monday']  # förenklad exempelflagga

# Skapa X och y
X = df[['lat', 'lon', 'hour']]  # förenklat exempel
y = df['num_crimes']

# Dela upp i träning/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Träna modell
model = LinearRegression()
model.fit(X_train, y_train)

# Utvärdera
print(f"R^2-score: {model.score(X_test, y_test):.2f}")