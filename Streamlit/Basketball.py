
import requests
import os
import shutil
from google.colab import files

years = list(range(1991,2022))

url_start = "https://www.basketball-reference.com/awards/awards_{}.html"

for year in years:
    url = url_start.format(year)

    data = requests.get(url)
    with open("awards_{}.html".format(year), "w+") as f:
        f.write(data.text)

from bs4 import BeautifulSoup

import pandas as pd

# Initialize an empty list to hold all the DataFrames
all_mvp_data = []

# Loop through each year from 1991 to 2021
for year in range(1991, 2022):  # 2022 is exclusive
    # Assuming you have a way to get the mvp_table for each year
    # For example, if mvp_table is loaded from a file:
    # mvp_table = pd.read_html("awards_{}.html".format(year))[0]
    # Or, if you have the HTML content in a variable already, use that

    mvp_df = pd.read_html(str(mvp_table))[0]  # Replace str(mvp_table) with your actual data source
    mvp_df["Year"] = year  # Add a 'Year' column
    all_mvp_data.append(mvp_df)  # Append the DataFrame to the list

# Concatenate all DataFrames into one
combined_mvp_data = pd.concat(all_mvp_data, ignore_index=True)

combined_mvp_data
combined_mvp_data.to_csv('combined_mvp_data_1991_2021.csv', index=False)