

import matplotlib
matplotlib.use("Agg")  # Use the "Agg" backend for Matplotlib


#1Trends in Performance Metrics Over Time: Line graphs showing the average of key stats like points, assists, and rebounds per year.
#2Player Appearances and Dominance: A bar chart to display the frequency of appearances in the top 10 MVP candidates list for the most frequent candidates.
#3Team Success Correlation: A scatter plot showing the relationship between team success (win-loss record) and MVP candidacy.
#4Positional Analysis Over Time: A line graph showing the distribution of MVP candidates by position over the years.
#5Age Distribution of MVP Candidates: A histogram to analyze the age distribution of MVP candidates.

import matplotlib.pyplot as plt

# Load the CSV data into a DataFrame
df = pd.read_csv('/Users/christiancheung/Documents/Stats386/Streamlit/combined_mvp_data_1991_2021.csv')

# Group by 'Year' and calculate the mean of 'PTS' for each year
average_stats_per_year = mean_pts = df.groupby('Year')['AST'].mean()

print(average_stats_per_year)
df.info()

def univariate_stats(df,gen_charts):
  #Step 1 generate output dataframe
  import pandas as pd
  import seaborn as sns
  from matplotlib import pyplot as plt
  import matplotlib.pyplot as plt

  output_df = pd.DataFrame(columns=['Count','Null' ,'Unique', 'Type', 'Min', 'Max', '25%', '50%', '75%', 'Mean', 'Mode', 'Std', 'Skew', 'Kurt'])

  #Step 2 iterate through columns in df
  for col in df:

  #Step 3 determine if column is numeric
    if pd.api.types.is_numeric_dtype(df[col]):
    #Step 4a generate stats for numeric col
      oRow = [df[col].count(),df[col].isna().sum(),df[col].nunique(),df[col].dtype,df[col].min(),df[col].max(),
      df[col].quantile(0.25),df[col].median(),df[col].quantile(0.75),round(df[col].mean(),2),
      round(df[col].mode()[0],2),df[col].std(),df[col].skew(),df[col].kurt()]
      output_df.loc[col] = oRow
      #Step 5a generate hist plot if boolean is true
      if gen_charts:
        str_text = 'Kurtosis is ' + str(df[col].kurt()) +'\n' + 'Skew is ' + str(df[col].skew())
        plt.text(max(df[col]) + (max(df[col])/10), 54,str_text)
        sns.histplot(data =df, x = col)
        plt.show()
    else:
      #Step 4b generate stats for categorical col
      oRow = [df[col].count(), df[col].isna().sum(), df[col].nunique(),df[col].dtype,'-','-',
      '-','-','-','-',
      df[col].mode()[0],'-','-','-']

      #Step 5b generate count plot if boolean is true
      if gen_charts:
        plt.xticks(rotation =45)
        sns.countplot(data =df, x = col, palette="Greens_d",
                      order=df[col].value_counts().iloc[:6].index)
        plt.show()
    #Step 6 add row to dataframe
    output_df.loc[col] = oRow

  #step 7 output df
  return output_df

df_filtered = df.drop(columns=['Age' ,'PTS'])

univariate_stats(df_filtered, True)