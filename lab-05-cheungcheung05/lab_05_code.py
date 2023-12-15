# # Lab 5

# ## Part 1: Flights
# %%
import pandas as pd
import numpy as np

# %%
base_url = 'https://github.com/esnt/Data/raw/main/Flights/'
flights = pd.read_csv(base_url + 'flights.csv')
planes = pd.read_csv(base_url + 'planes.csv')
weather = pd.read_csv(base_url + 'weather.csv')
airlines = pd.read_csv(base_url + 'airlines.csv')


# ## 1. Number of entries
# %%
observations = len(flights)
print(observations)
# ## 2. Variable with most missing values
# %%

missing_values = flights.isna().sum()

sorted_missing_values = missing_values.sort_values(ascending = False)

most_missing_variables = sorted_missing_values.index[:2]

missing_values_count = sorted_missing_values.iloc[:2]

print(missing_values_count)


# ## 3. Average airtime
# %%
mean_air_time = flights['air_time'].mean()

print(mean_air_time)


# ## 4. Unique destinations
# %%
unique_values_count = flights.nunique()
print(unique_values_count)

# ## 5. Five most common destinations
# %%

dest_count = flights['dest'].value_counts().head(5)
print(dest_count)

# ## 6. Max distance
# %%
max_distance = flights['distance'].max()
print(max_distance)

# ## 7. Number flight that flew max distance
# %%
num_flights_max_distance = (flights['distance'] == max_distance).sum()
print(num_flights_max_distance)

#max_distance_rows = flights.loc[flights['distance'].idxmax()]
#flights_with_max_distance = len(max_distance_rows)
#flights_with_max_distance
# ## 8. Create cancelled variable
# %%
flights['cancelled'] = flights['air_time'].isnull().astype(int)
#a. 
cancelled_by_month = flights.groupby(flights['month'].astype(str).str.zfill(2))['cancelled'].mean()

highest_cancelled_month = cancelled_by_month.idxmax()
print(highest_cancelled_month)
#b.
highest_cancelled_proportion = cancelled_by_month.max()
print(highest_cancelled_proportion)

#c.
cancelled_by_origin = flights.groupby('origin')['cancelled'].mean()
lowest_cancelled_origin = cancelled_by_origin.idxmin()
print(lowest_cancelled_origin)

#d. 
lowest_cancelled_proportion = cancelled_by_origin.min()
print(lowest_cancelled_proportion)


# ## 9. Date with largest avg arrival delay
# %%
flights['date'] = pd.to_datetime(flights[['year', 'month', 'day']])
largest_delay = flights.groupby('date')['arr_delay'].mean().idxmax()
largest_delay.strftime('%m/%d')

average_largest_delay = flights.groupby('date')['arr_delay'].mean().max()
print(average_largest_delay)

# ## 10. Tail numbers
# %%
#a.
most_common_tailnum = flights['tailnum'].mode().iloc[0]
print(most_common_tailnum)

#b. 
count_most_common_tailnum = flights[flights['tailnum'] == most_common_tailnum].shape[0]
print(count_most_common_tailnum)

#c.
average_distance = flights[flights['tailnum'] == most_common_tailnum]['distance'].mean()
print(average_distance)

#d.
most_common_route = flights[flights['tailnum'] == most_common_tailnum].groupby(['origin', 'dest']).size().idxmax()
formatted_most_common_route = "-".join(most_common_route)
print(formatted_most_common_route)

# ## 11. Missing mfg year
# %%
manufacturer_na = planes['year'].isna().sum()
print(manufacturer_na)

# ## 12. Oldest plane
# %%

#a.
oldest_plane = planes.loc[planes['year'].idxmin(), ['tailnum']]
print(oldest_plane)

#b. 
old_tail = planes.loc[planes['year'].idxmin(), 'tailnum']
current_year = 2013 
current_year - planes.loc[planes['tailnum'] == old_tail, 'year'].iloc[0]



# ## 13. Create DF with carrier code, carrier name, and avg plane age for the carrier
# %%
merged_df = pd.merge(flights, planes[['tailnum', 'year']], how='left', left_on='tailnum', right_on='tailnum')
merged_df['plane_age'] = merged_df['year_x'] - merged_df['year_y']
avg_plane_age_by_carrier = merged_df.groupby('carrier')['plane_age'].mean().reset_index()
carrier_names = {
    'DL': 'Delta Air Lines Inc.',
    'AA': 'American Airlines Inc.',
    'UA': 'United Airlines Inc.',
    'B6': 'JetBlue Airways',
    'AS': 'Alaska Airlines Inc.',
    'NK': 'Spirit Air Lines',
    'WN': 'Southwest Airlines Co.',
    'F9': 'Frontier Airlines Inc.',
    'HA': 'Hawaiian Airlines Inc.',
    'VX': 'Virgin America',
    'OO': 'SkyWest Airlines Inc.',
    'EV': 'ExpressJet Airlines Inc.',
    'MQ': 'Envoy Air',
    'US': 'US Airways Inc.',
    'FL': 'AirTran Airways Corporation'
}
avg_plane_age_by_carrier['carrier_name'] = avg_plane_age_by_carrier['carrier'].map(carrier_names)
final_df = avg_plane_age_by_carrier[['carrier_name', 'carrier', 'plane_age']].sort_values(by='plane_age').reset_index(drop=True)
final_df
# final_df.to_csv('avg_carrier_age.csv', index=None)

# ## 14. Correlation between no. of seats and avg distance
# %%
merged = pd.merge(flights, planes[['tailnum', 'seats']], how='left', left_on='tailnum', right_on='tailnum')

avg_distance_by_plane = merged_df.groupby('tailnum')['distance'].mean().reset_index()

planes_df = pd.merge(planes, avg_distance_by_plane, how='left', left_on='tailnum', right_on='tailnum')

planes_df['distance'].fillna(0, inplace=True)

correlation_coefficient = planes_df['seats'].corr(planes_df['distance'])
correlation_coefficient

# ## 15. Days with precip
# %%
weather['precip'] = pd.to_numeric(weather['precip'].replace('T', 0)) 
weather['time_hour'] = pd.to_datetime(weather['time_hour']) 
weather_2013 = weather[weather['time_hour'].dt.year == 2013]
precipitation_days = weather_2013[weather_2013['precip'] > 0]['time_hour'].dt.date.nunique()

precipitation_days

# ## 16. Correlation between precip and avg delay
# %%
#flights['time_hour'] = pd.to_datetime(flights['time_hour'])
#weather['time_hour'] = pd.to_datetime(weather['time_hour'])

#merged_data = pd.merge(flights, weather, on='time_hour', how='inner')
#precipitation_days = merged_data[merged_data['precip'] > 0]['time_hour'].dt.date.unique()
#merged_data_precip = merged_data[merged_data['time_hour'].dt.date.isin(precipitation_days)]
#merged_data_precip = merged_data_precip.dropna(subset=['precip', 'dep_delay'])
#correlation_coefficient_precip = merged_data_precip['precip'].corr(merged_data_precip['dep_delay'])

#print(correlation_coefficient_precip)


# ## Part 2:  US Cities

# ## 17. Table on webpage
# %%
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
url = "https://en.wikipedia.org/wiki/List_of_United_States_cities_by_population"

tables = pd.read_html(url) 
df = tables[4]
df
len(tables)


# ## 18. Index of table
# %%
tables[4]
# ## 19. Percent of population in cities
# %%
us_pop = 331449281
cities_population = df["2020 census"].sum()
percent_cities = cities_population / us_pop
percent_cities
# ## 20.  Percent of land area in cities
# %%
for idx, table in enumerate(tables):
    if all(col in table.columns for col in df):
        table_index = idx
        break

df["2020 land area"] = df["2020 land area"].astype(str).str.replace(',', '').str.extract('(\d+.\d+)').astype(float)

land_area = 3800000  
cities_land_area = df["2020 land area"].sum()
cities_land_area / land_area 

# %%
