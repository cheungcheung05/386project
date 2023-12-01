
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.title('Popular Names')

url = 'https://github.com/esnt/Data/raw/main/Names/popular_names.csv'

df = pd.read_csv(url)

selected_name = st.text_input('Enter a name', 'John') #First field is prompt, second field is default

name_df = df[df['name'] == selected_name]

if name_df.empty:
    st.write('Name not found :(')
else:
    fig = px.line(name_df, x = 'year', y = 'n', color = 'sex',
                  color_discrete_sequence = ['blue','red'])
    st.plotly_chart(fig)

selected_year = st.selectbox('Select a year', df['year'].unique())

year_df = df[df['year'] == selected_year]
fnames = year_df[year_df['sex'] == 'F'].sort_values(by = 'n', ascending = False).head(5)
mnames = year_df[year_df['sex'] == 'M'].sort_values(by = 'n', ascending = False).head(5)

top_names = pd.concat([fnames, mnames], axis = 0) 

st.write(f'Top names in {selected_year}:')
st.dataframe(top_names)