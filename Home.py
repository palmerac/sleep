import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import datetime as dt
import os
import glob

plt.style.use('ggplot')
st.set_page_config(layout="wide")

# Read files
uploaded_file = st.sidebar.file_uploader("Upload CSV file from AutoSleep", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    # Get all csv files in the folder
    csv_files = glob.glob('csvs/*.csv')

    # Read all csv files and concatenate them
    df = pd.concat([pd.read_csv(f) for f in csv_files])
    # Remove duplicate rows
    df = df.drop_duplicates()

# Drop Unneccesary rows
df = df.drop(['fellAsleepIn', 'SpO2Avg', 'SpO2Min', 'SpO2Max', 'respAvg', 'respMin', 'respMax',
              'tags', 'notes', 'asleepAvg7','efficiencyAvg7', 'qualityAvg7', 'deepAvg7', 'sleepBPMAvg7', 
              'dayBPMAvg7', 'wakingBPMAvg7', 'hrvAvg7','sleepHRVAvg7'], axis=1)

# Datetime conversions
df['ISO8601'] = pd.to_datetime(df['ISO8601'])
df['fromDate'] = pd.to_datetime(df['fromDate'])
df['toDate'] = pd.to_datetime(df['toDate'])
df['bedtime'] = pd.to_datetime(df['bedtime']).dt.time
df['waketime'] = pd.to_datetime(df['waketime']).dt.time
df['inBed'] = pd.to_datetime(df['inBed'], format='%H:%M:%S').dt.hour + (pd.to_datetime(df['inBed'], format='%H:%M:%S').dt.minute / 60)
df['awake'] = pd.to_datetime(df['awake'], format='%H:%M:%S').dt.hour  + (pd.to_datetime(df['awake'], format='%H:%M:%S').dt.minute / 60)
df['asleep'] = pd.to_datetime(df['asleep'], format='%H:%M:%S').dt.hour  + (pd.to_datetime(df['asleep'], format='%H:%M:%S').dt.minute / 60)
df['quality'] = pd.to_datetime(df['quality'], format='%H:%M:%S').dt.hour  + (pd.to_datetime(df['quality'], format='%H:%M:%S').dt.minute / 60)
df['deep'] = pd.to_datetime(df['deep'], format='%H:%M:%S').dt.hour  + (pd.to_datetime(df['deep'], format='%H:%M:%S').dt.minute / 60)

df = df[df['asleep'] <= 15]


# Elapsed days and first/last
daysPassed = (df['fromDate'].max() - df['fromDate'].min()).days
first_day = df['fromDate'].min().strftime('%Y-%m-%d')
last_day = df['fromDate'].max().strftime('%Y-%m-%d')

# Totals
tot_sleeptime = df['asleep'].sum()
tot_sleeptime_pct = round(tot_sleeptime / (len(df) * 24)*100, 2)


# Find the percentage of days tracked
percentage_tracked = round((len(df) / daysPassed) * 100,2)

from datetime import datetime
from dateutil.relativedelta import relativedelta

timeElapsed = relativedelta(df['fromDate'].max(), df['fromDate'].min())
timeElapsed = f"{timeElapsed.years} years, {timeElapsed.months} months, {timeElapsed.days} days"

monthsMissed = (daysPassed - len(df)) // 30
daysMissed = (daysPassed - len(df)) % 30

# Rolling values
cols_to_roll = ['asleep', 'quality', 'deep', 'efficiency', 'sleepBPM', 'wakingBPM', 'hrv', 'sleepHRV']
periods = [7,15,30,60, 90]

for col in cols_to_roll:
    for period in periods:
        df[f'{col}Roll{period}'] = df[col].rolling(window=period).mean()

df['qual/asleep'] = df['quality'] / df['asleep']
df['deep/asleep'] = df['deep'] / df['asleep']

# Streamlit
st.title('Sleep Data Dashboard')

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Summary', 'Charts','Rolling Window Charts', 'Boxplots', 'Correlation Matrix'])

with tab1:
    st.header('Summary')
    years = int(tot_sleeptime // 8760)
    months = int((tot_sleeptime % 8760) // 730)
    days = int(((tot_sleeptime % 8760) % 730) // 24)
    hours = int(((tot_sleeptime % 8760) % 730) % 24)
    minutes = int((tot_sleeptime % 1) * 60)
    time_slept = f'Total Time Slept: '
    if years != 0:
        time_slept += f'{years} years, '
    if months != 0:
        time_slept += f'{months} months, '
    if days != 0:
        time_slept += f'{days} days, '
    if hours != 0:
        time_slept += f'{hours} hours, '
    if minutes != 0:
        time_slept += f'{minutes} minutes'
    st.text(time_slept)
    st.text(f"Time Slept as % of Time Elapsed: {tot_sleeptime_pct}%")
    st.text("------------------------------------------------------------")
    st.text(f"First day tracked: {first_day}")
    st.text(f"Most recent day tracked: {last_day}")
    st.text("------------------------------------------------------------")
    st.text(f"Percentage of days tracked: {percentage_tracked}%")
    st.text(f'Time Elapsed: {timeElapsed}')
    st.text(f'Time Missed: {monthsMissed} months, {daysMissed} days')
    st.text("------------------------------------------------------------")

with tab2:
    st.header('Charts')
    start_date_key = "start_date_input"  # Unique key for the start date input widget
    start_date = pd.to_datetime(st.date_input("Start Date", value=df['fromDate'].min().date(), key=start_date_key))

    end_date_key = "end_date_input"  # Unique key for the start date input widget
    end_date = pd.to_datetime(st.date_input("End Date", value=df['fromDate'].max().date(), key=end_date_key))
    # Weekday Chart
    with st.expander('Average Sleep and Sleep BPM by Weekday'):
        st.text("Weekends are generally delayed one day due to falling asleep after midnight (ex. Sunday value is Saturday night)")
        df_filtered = df[(df['fromDate'] >= start_date) & (df['fromDate'] <= end_date)]
        df_filtered['weekday'] = df_filtered['fromDate'].dt.day_name()
        df_filtered['weekday'] = pd.Categorical(df_filtered['weekday'], categories=['Monday', 'Tuesday', 'Wednesday',
                                                                                     'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)
        df_filtered['asleep'] = df_filtered['asleep'].astype(float)
        df_filtered['sleepBPM'] = df_filtered['sleepBPM'].astype(float)

        df_filtered_weekday = df_filtered.groupby('weekday').agg({'asleep': 'mean', 'sleepBPM': 'mean'}).reset_index()

        st.set_option('deprecation.showPyplotGlobalUse', False)
        fig, ax1 = plt.subplots(figsize=(12,6))

        color = 'tab:red'
        ax1.set_xlabel('Weekday')
        ax1.set_ylabel('Average Asleep Time', color=color)
        ax1.bar(df_filtered_weekday['weekday'], df_filtered_weekday['asleep'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(round(0.975*df_filtered_weekday['asleep'].min() / 0.1) * 0.1, round(1.025*df_filtered_weekday['asleep'].max() / 0.1) * 0.1)  

        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Average Sleep BPM', color=color)  
        ax2.plot(df_filtered_weekday['weekday'], df_filtered_weekday['sleepBPM'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  
        plt.title('Average Asleep Time and Sleep BPM by Weekday')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    with st.expander("Average Sleep and Sleep BPM by Month"):
        df_filtered['month'] = df_filtered['fromDate'].dt.month_name()
        df_filtered['month'] = pd.Categorical(df_filtered['month'], categories=['January', 'February', 'March', 'April', 'May', 'June',
                                                                                 'July', 'August', 'September', 'October', 'November', 'December'], ordered=True)
        df_filtered['asleep'] = df_filtered['asleep'].astype(float)
        df_filtered['sleepBPM'] = df_filtered['sleepBPM'].astype(float)

        df_filtered_month = df_filtered.groupby('month').agg({'asleep': 'mean', 'sleepBPM': 'mean'}).reset_index()
        df_filtered_month['datapoints'] = df_filtered.groupby('month').size().values

        fig, ax1 = plt.subplots(figsize=(12,6))

        color = 'tab:red'
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Average Asleep Time', color=color)
        ax1.bar(df_filtered_month['month'], df_filtered_month['asleep'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(round(0.975*df_filtered_month['asleep'].min() / 0.25) * 0.25, round(1.025*df_filtered_month['asleep'].max() / 0.25) * 0.25)

        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Average Sleep BPM', color=color)  
        ax2.plot(df_filtered_month['month'], df_filtered_month['sleepBPM'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  
        plt.title('Average Asleep Time and Sleep BPM by Month')
        plt.xticks(rotation=45)
        st.pyplot(fig)

with tab3:
    st.header('Rolling Window Averages')
    start_date = pd.to_datetime(st.date_input("Start Date", value=df['fromDate'].min().date()))
    end_date = pd.to_datetime(st.date_input("End Date", value=df['fromDate'].max().date()))
    df_filtered = df[(df['fromDate'] >= start_date) & (df['fromDate'] <= end_date)]
    period = st.selectbox('Select Period', [7, 15, 30, 60, 90])

    st.set_option('deprecation.showPyplotGlobalUse', False)
    with st.expander("Time Asleep"):
        fig, ax = plt.subplots(figsize=(18,6))
        ax.plot(df_filtered['fromDate'], df_filtered[f'asleepRoll{period}'])
        ax.set_title(f'{period}-day Rolling Average Sleep')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sleep BPM')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with st.expander("Sleep BPM"):
        fig1, ax1 = plt.subplots(figsize=(16,6))
        ax1.plot(df_filtered['fromDate'], df_filtered[f'sleepBPMRoll{period}'])
        ax1.set_title(f'{period}-day Rolling Average Sleep BPM')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sleep BPM')
        plt.xticks(rotation=45)
        st.pyplot(fig1)

    with st.expander("Efficiency"):
        fig1, ax1 = plt.subplots(figsize=(16,6))
        ax1.plot(df_filtered['fromDate'], df_filtered[f'efficiencyRoll{period}'])
        ax1.set_title(f'{period}-day Rolling Average Efficiency')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Efficiency')
        plt.xticks(rotation=45)
        st.pyplot(fig1)

    with st.expander("Sleep HRV"):
        fig1, ax1 = plt.subplots(figsize=(16,6))
        ax1.plot(df_filtered['fromDate'], df_filtered[f'sleepHRVRoll{period}'])
        ax1.set_title(f'{period}-day Rolling Average Sleep HRV')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sleep HRV')
        plt.xticks(rotation=45)
        st.pyplot(fig1)

    with st.expander("Raw Data"):
        st.dataframe(df_filtered)

with tab4:
    start_date_box = pd.to_datetime(st.date_input("Start Date for Boxplot", value=df['fromDate'].min().date()))
    end_date_box = pd.to_datetime(st.date_input("End Date for Boxplot", value=df['fromDate'].max().date()))
    df_box = df[(df['fromDate'] >= start_date_box) & (df['fromDate'] <= end_date_box)]
    quantv = st.selectbox('Select quantile value of outliers to remove (from each column)', 
                          options=[0,0.001, 0.005, 0.01, 0.02], index=0)
    df_box = df[['asleep', 'deep', 'sleepHRV', 'sleepBPM', 'wakingBPM', 'efficiency']]
    
    if quantv != 0:
        for col in df_box.columns:
            df_box = df_box[(df_box[col] > df_box[col].quantile(quantv)) & (df_box[col] < df_box[col].quantile(1-quantv))]

    st.text(f"Percentage of outliers removed: {round(((len(df) - len(df_box)) / len(df)) * 100, 2)}%")

    # Boxplots
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig, axes = plt.subplots(nrows=len(df_box.columns)//2, ncols=2, figsize=(16, 16))
    plt.suptitle('Boxplots', size=20)
    for i, col in enumerate(df_box.columns):
        sns.boxplot(x=col, data=df_box, ax=axes[i//2, i%2])
        axes[i//2, i%2].set_title(f'{col}')
    plt.tight_layout()
    st.pyplot(fig)

with tab5:
    correlation_matrix = round(df[[col for col in df.columns if 'Roll' not in col and '/' not in col]].corr(),4)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)
