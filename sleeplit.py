import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import datetime as dt

plt.style.use('ggplot')

# Read files
df1 = pd.read_csv('AutoSleep-20200124-to-20231109.csv')
df2 = pd.read_csv('AutoSleep-20231018-to-20240415.csv')

# Filter out rows from df2 that are already in df1
df2 = df2[~df2['fromDate'].isin(df1['fromDate'])]

# Concatenate the DataFrames
df = pd.concat([df1, df2])

# Drop Unneccesart rows
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

# Count how many days since the date of the first record in df until today
daysPassed = (df['fromDate'].max() - df['fromDate'].min()).days

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
st.set_page_config(layout="wide")
st.title('Sleep Data Dashboard')

tab1, tab2, tab3 = st.tabs(['Summary', 'Rolling Window Means', 'Boxplots'])

with tab1:
    st.header('Summary Statistics')
    st.text(f"Percentage of days tracked: {percentage_tracked}%")
    st.text(f'Time Elapsed: {timeElapsed}')
    st.text(f'Time Missed: {monthsMissed} months, {daysMissed} days')
    # st.text()


with tab2:
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

with tab3:
    start_date_box = pd.to_datetime(st.date_input("Start Date for Boxplot", value=df['fromDate'].min().date()))
    end_date_box = pd.to_datetime(st.date_input("End Date for Boxplot", value=df['fromDate'].max().date()))
    df_box = df[(df['fromDate'] >= start_date_box) & (df['fromDate'] <= end_date_box)]
    
    quantv = st.number_input('Enter quantile value of outliers to remove', min_value=float(0), max_value=0.99, value=0.01, step=0.01)
    df_box = df[['asleep', 'deep', 'sleepHRV', 'sleepBPM', 'wakingBPM', 'efficiency']]
    
    
    if quantv != 0:
        for col in df_box.columns:
            df_box = df_box[(df_box[col] > df_box[col].quantile(quantv)) & (df_box[col] < df_box[col].quantile(1-quantv))]

    st.text(f"Percentage of df included: {round(len(df_box) / len(df)*100, 2)}%")

    # Boxplots
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig, axes = plt.subplots(nrows=len(df_box.columns)//2, ncols=2, figsize=(16, 16))
    plt.suptitle('Boxplots', size=20)
    for i, col in enumerate(df_box.columns):
        sns.boxplot(x=col, data=df_box, ax=axes[i//2, i%2])
        axes[i//2, i%2].set_title(f'{col}')
    plt.tight_layout()
    st.pyplot(fig)
