import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import datetime as dt
from scipy import stats

plt.style.use("ggplot")
st.set_page_config(layout="wide")

st.title("AutoSleep Dashboard")
st.sidebar.markdown("Made with ❤️ by [palmerac](https://github.com/palmerac)")

# Read files
st.sidebar.markdown("To run with your own AutoSleep data, follow these steps:")
st.sidebar.markdown("1. On AutoSleep app and go to Settings")
st.sidebar.markdown("2. Export --> History")
st.sidebar.markdown("3. Select Dates")
st.sidebar.markdown("4. Download File")
st.sidebar.markdown("4. Upload File in box below")

# uploaded_file = st.sidebar.file_uploader("Upload Box", type=['csv'])

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
# else:
#     # Get all csv files in the folder
#     csv_files = ['csvs/AutoSleep-20200124-to-20231109.csv', 'csvs/AutoSleep-20231018-to-20240415.csv']

#     # Read all csv files and concatenate them
#     df = pd.concat([pd.read_csv(f) for f in csv_files])
#     # Remove duplicate rows
#     df = df.drop_duplicates(subset='fromDate')


@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        import os

        csv_files = [
            os.path.join("csvs", f) for f in os.listdir("csvs") if f.endswith(".csv")
        ]
        df = pd.concat([pd.read_csv(f) for f in csv_files])
        df = df.drop_duplicates(subset="fromDate")
    return df


uploaded_file = st.sidebar.file_uploader("Upload Box", type=["csv"])
df = load_data(uploaded_file)

# Drop Unneccesary rows
df = df.drop(
    [
        "ISO8601",
        "fellAsleepIn",
        "SpO2Avg",
        "SpO2Min",
        "SpO2Max",
        "respAvg",
        "respMin",
        "respMax",
        "tags",
        "notes",
        "asleepAvg7",
        "efficiencyAvg7",
        "qualityAvg7",
        "deepAvg7",
        "sleepBPMAvg7",
        "dayBPMAvg7",
        "wakingBPMAvg7",
        "hrvAvg7",
        "sleepHRVAvg7",
    ],
    axis=1,
)

# Datetime conversions
df["fromDate"] = pd.to_datetime(df["fromDate"])
df["toDate"] = pd.to_datetime(df["toDate"])
df["bedtime"] = pd.to_datetime(df["bedtime"]).dt.time
df["waketime"] = pd.to_datetime(df["waketime"]).dt.time
df["inBed"] = pd.to_datetime(df["inBed"], format="%H:%M:%S").dt.hour + (
    pd.to_datetime(df["inBed"], format="%H:%M:%S").dt.minute / 60
)
df["awake"] = pd.to_datetime(df["awake"], format="%H:%M:%S").dt.hour + (
    pd.to_datetime(df["awake"], format="%H:%M:%S").dt.minute / 60
)
df["asleep"] = pd.to_datetime(df["asleep"], format="%H:%M:%S").dt.hour + (
    pd.to_datetime(df["asleep"], format="%H:%M:%S").dt.minute / 60
)
df["quality"] = pd.to_datetime(df["quality"], format="%H:%M:%S").dt.hour + (
    pd.to_datetime(df["quality"], format="%H:%M:%S").dt.minute / 60
)
df["deep"] = pd.to_datetime(df["deep"], format="%H:%M:%S").dt.hour + (
    pd.to_datetime(df["deep"], format="%H:%M:%S").dt.minute / 60
)

df = df[df["asleep"] <= 15]

# Elapsed days and first/last
daysPassed = (df["fromDate"].max() - df["fromDate"].min()).days
first_day = df["fromDate"].min().strftime("%Y-%m-%d")
last_day = df["fromDate"].max().strftime("%Y-%m-%d")

# Totals
tot_sleeptime = df["asleep"].sum()
tot_sleeptime_pct = round(tot_sleeptime / (len(df) * 24) * 100, 2)
tot_bedtime = df["inBed"].sum()
tot_bedtime_pct = round(tot_bedtime / (len(df) * 24) * 100, 2)

# Find the percentage of days tracked
percentage_tracked = round((len(df) / daysPassed) * 100, 2)

from datetime import datetime
from dateutil.relativedelta import relativedelta

timeElapsed = relativedelta(df["fromDate"].max(), df["fromDate"].min())
timeElapsed = (
    f"{timeElapsed.years} years, {timeElapsed.months} months, {timeElapsed.days} days"
)

monthsMissed = (daysPassed - len(df)) // 30
daysMissed = (daysPassed - len(df)) % 30

# Rolling values
cols_to_roll = [
    "asleep",
    "quality",
    "deep",
    "efficiency",
    "sleepBPM",
    "wakingBPM",
    "hrv",
    "sleepHRV",
]
periods = [15, 30, 60, 90, 180]

for col in cols_to_roll:
    for period in periods:
        df[f"{col}Roll{period}"] = df[col].rolling(window=period).mean()

df["qual/asleep"] = df["quality"] / df["asleep"]
df["deep/asleep"] = df["deep"] / df["asleep"]
df["TotalBeats"] = df["asleep"] * df["sleepBPM"] * 60

# Streamlit
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Summary",
        "Charts",
        "Moving Average Charts",
        "Boxplots",
        "Histograms",
        "Correlation Matrix",
    ]
)

with tab1:
    st.header("Summary")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        avg_sleep_hours = int(df["asleep"].mean())
        avg_sleep_minutes = round((df["asleep"].mean() - avg_sleep_hours) * 60)
        median_sleep_hours = int(df["asleep"].median())
        median_sleep_minutes = round((df["asleep"].median() - median_sleep_hours) * 60)
        mode_sleep_hours = int(df["asleep"].mode().iloc[0])
        mode_sleep_minutes = round(
            (df["asleep"].mode().iloc[0] - mode_sleep_hours) * 60
        )
        mode_sleep_hours_count = df["asleep"].value_counts().max()
        max_sleep_hours = int(df["asleep"].max())
        max_sleep_minutes = round((df["asleep"].max() - max_sleep_hours) * 60)
        min_sleep_hours = int(df["asleep"].min())
        min_sleep_minutes = round((df["asleep"].min() - min_sleep_hours) * 60)

        st.markdown("#### Sleep Time")
        st.markdown(f"Average: {avg_sleep_hours}:{avg_sleep_minutes}")
        st.markdown(f"Median: {median_sleep_hours}:{median_sleep_minutes}")
        st.markdown(
            f"Mode: {mode_sleep_hours}:{mode_sleep_minutes} ({mode_sleep_hours_count})"
        )
        st.markdown(f"Max: {max_sleep_hours}:{max_sleep_minutes}")
        st.markdown(f"Min: {min_sleep_hours}:{min_sleep_minutes}")
        ## Add Var?

    with col2:
        avg_inbed_hours = int(df["inBed"].mean())
        avg_inbed_minutes = round((df["inBed"].mean() - avg_inbed_hours) * 60)
        median_inbed_hours = int(df["inBed"].median())
        median_inbed_minutes = round((df["inBed"].median() - median_inbed_hours) * 60)
        mode_inbed_hours = int(df["inBed"].mode().iloc[0])
        mode_inbed_minutes = round((df["inBed"].mode().iloc[0] - mode_inbed_hours) * 60)
        mode_inbed_hours_count = df["inBed"].value_counts().max()
        max_inbed_hours = int(df["inBed"].max())
        max_inbed_minutes = round((df["inBed"].max() - max_inbed_hours) * 60)
        min_inbed_hours = int(df["inBed"].min())
        min_inbed_minutes = round((df["inBed"].min() - min_inbed_hours) * 60)

        st.markdown("#### In Bed Time")
        st.markdown(f"Average: {avg_inbed_hours}:{avg_inbed_minutes}")
        st.markdown(f"Median: {median_inbed_hours}:{median_inbed_minutes}")
        st.markdown(
            f"Mode: {mode_inbed_hours}:{mode_inbed_minutes} ({mode_inbed_hours_count})"
        )
        st.markdown(f"Max: {max_inbed_hours}:{max_inbed_minutes}")
        st.markdown(f"Min: {min_inbed_hours}:{min_inbed_minutes}")

    with col3:
        st.markdown("#### Sleep BPM")
        st.markdown(f"Average: {round(df['sleepBPM'].mean(),1)}")
        st.markdown(f"Median: {round(df['sleepBPM'].median(),1)}")
        st.markdown(
            f"Mode: {round(df['sleepBPM'].mode().iloc[0],1)} ({df['sleepBPM'].value_counts().max()})"
        )
        st.markdown(f"Max: {round(df['sleepBPM'].max(),1)}")
        st.markdown(f"Min: {round(df['sleepBPM'].min(),1)}")

    with col4:
        st.markdown("#### Waking BPM")
        st.markdown(f"Average: {round(df['wakingBPM'].mean(),1)}")
        st.markdown(f"Median: {round(df['wakingBPM'].median(),1)}")
        st.markdown(
            f"Mode: {round(df['wakingBPM'].mode().iloc[0],1)} ({df['wakingBPM'].value_counts().max()})"
        )
        st.markdown(f"Max: {round(df['wakingBPM'].max(),1)}")
        st.markdown(f"Min: {round(df['wakingBPM'].min(),1)}")

    with col5:
        st.markdown("#### Sleep HRV")
        st.markdown(f"Average: {round(df['sleepHRV'].mean(),1)}")
        st.markdown(f"Median: {round(df['sleepHRV'].median(),1)}")
        st.markdown(
            f"Mode: {round(df['sleepHRV'].mode().iloc[0],1)} ({df['sleepHRV'].value_counts().max()})"
        )
        st.markdown(f"Max: {round(df['sleepHRV'].max(),1)}")
        st.markdown(f"Min: {round(df['sleepHRV'].min(),1)}")

    with col6:
        st.markdown("#### Efficiency")
        st.markdown(f"Average: {round(df['efficiency'].mean(),1)}")
        st.markdown(f"Median: {round(df['efficiency'].median(),1)}")
        st.markdown(
            f"Mode: {round(df['efficiency'].mode().iloc[0],1)} ({df['efficiency'].value_counts().max()})"
        )
        st.markdown(f"Max: {round(df['efficiency'].max(),1)}")
        st.markdown(f"Min: {round(df['efficiency'].min(),1)}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"First day tracked: {first_day}")
        st.markdown(f"Most recent day tracked: {last_day}")
        st.markdown(f"Time Elapsed: {timeElapsed}")
        st.markdown(f"Time Missed: {monthsMissed} months, {daysMissed} days")
    with col2:
        st.markdown(f"Sleeps tracked: {len(df)}")
        st.markdown(f"Sleeps missed: {daysPassed - len(df)}")
        st.markdown(f"Percentage of days tracked: {percentage_tracked}%")
        st.markdown(f"Total Heartbeats: {df['TotalBeats'].sum().astype(int):,}")

    st.markdown("---")

    # Time Slept
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"Total Hours Slept: {round(tot_sleeptime,):,}h")
        st.markdown(f"% of day spent Sleeping: {tot_sleeptime_pct}%")

        years = int(tot_sleeptime // 8760)
        months = int((tot_sleeptime % 8760) // 730)
        days = int(((tot_sleeptime % 8760) % 730) // 24)
        hours = int(((tot_sleeptime % 8760) % 730) % 24)
        minutes = int((((tot_sleeptime % 8760) % 730) % 24 - hours) * 60)
        time_slept = f"Total Time Slept: "
        if years != 0:
            time_slept += f"{years} years, "
        if months != 0:
            time_slept += f"{months} months, "
        if days != 0:
            time_slept += f"{days} days, "
        if hours != 0:
            time_slept += f"{hours} hours, "
        if minutes != 0:
            time_slept += f"{minutes} minutes"
        st.markdown(time_slept)

    with col2:
        st.markdown(f"Total Time in Bed: {round(tot_bedtime):,}h")
        st.markdown(f"% of day spent in Bed: {tot_bedtime_pct}%")
        years = int(tot_bedtime // 8760)
        months = int((tot_bedtime % 8760) // 730)
        days = int(((tot_bedtime % 8760) % 730) // 24)
        hours = int(((tot_bedtime % 8760) % 730) % 24)
        minutes = int((((tot_bedtime % 8760) % 730) % 24 - hours) * 60)
        time_inbed = f"Total Time in Bed: "
        if years != 0:
            time_inbed += f"{years} years, "
        if months != 0:
            time_inbed += f"{months} months, "
        if days != 0:
            time_inbed += f"{days} days, "
        if hours != 0:
            time_inbed += f"{hours} hours, "
        if minutes != 0:
            time_inbed += f"{minutes} minutes"
        st.markdown(time_inbed)

with tab2:
    start_date_key = "start_date_input"  # Unique key for the start date input widget
    start_date = pd.to_datetime(
        st.date_input(
            "Start Date", value=df["fromDate"].min().date(), key=start_date_key
        )
    )

    end_date_key = "end_date_input"  # Unique key for the start date input widget
    end_date = pd.to_datetime(
        st.date_input("End Date", value=df["fromDate"].max().date(), key=end_date_key)
    )
    # Weekday Chart
    st.text("Average Sleep and Sleep BPM by:")
    with st.expander("Weekday"):
        st.text(
            "Weekends are generally delayed one day due to falling asleep after midnight (ex. Sunday value is Saturday night)"
        )
        df_filtered = df[(df["fromDate"] >= start_date) & (df["fromDate"] <= end_date)]
        df_filtered["weekday"] = df_filtered["fromDate"].dt.day_name()
        df_filtered["weekday"] = pd.Categorical(
            df_filtered["weekday"],
            categories=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            ordered=True,
        )
        df_filtered["asleep"] = df_filtered["asleep"].astype(float)
        df_filtered["sleepBPM"] = df_filtered["sleepBPM"].astype(float)

        df_filtered_weekday = (
            df_filtered.groupby("weekday")
            .agg({"asleep": "mean", "sleepBPM": "mean"})
            .reset_index()
        )

        fig, ax1 = plt.subplots(figsize=(12, 6))

        color = "tab:red"
        ax1.set_xlabel("Weekday")
        ax1.set_ylabel("Average Asleep Time", color=color)
        ax1.bar(
            df_filtered_weekday["weekday"], df_filtered_weekday["asleep"], color=color
        )
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_ylim(
            round(0.975 * df_filtered_weekday["asleep"].min() / 0.1) * 0.1,
            round(1.025 * df_filtered_weekday["asleep"].max() / 0.1) * 0.1,
        )

        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("Average Sleep BPM", color=color)
        ax2.plot(
            df_filtered_weekday["weekday"], df_filtered_weekday["sleepBPM"], color=color
        )
        ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()
        plt.title("Average Asleep Time and Sleep BPM by Weekday")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with st.expander("Month"):
        df_filtered["month"] = df_filtered["fromDate"].dt.month_name()
        df_filtered["month"] = pd.Categorical(
            df_filtered["month"],
            categories=[
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ],
            ordered=True,
        )
        df_filtered["asleep"] = df_filtered["asleep"].astype(float)
        df_filtered["sleepBPM"] = df_filtered["sleepBPM"].astype(float)

        df_filtered_month = (
            df_filtered.groupby("month")
            .agg({"asleep": "mean", "sleepBPM": "mean"})
            .reset_index()
        )
        df_filtered_month["datapoints"] = df_filtered.groupby("month").size().values

        fig, ax1 = plt.subplots(figsize=(12, 6))

        color = "tab:red"
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Average Asleep Time", color=color)
        ax1.bar(df_filtered_month["month"], df_filtered_month["asleep"], color=color)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_ylim(
            round(0.975 * df_filtered_month["asleep"].min() / 0.25) * 0.25,
            round(1.025 * df_filtered_month["asleep"].max() / 0.25) * 0.25,
        )

        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("Average Sleep BPM", color=color)
        ax2.plot(df_filtered_month["month"], df_filtered_month["sleepBPM"], color=color)
        ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()
        plt.title("Average Asleep Time and Sleep BPM by Month")
        plt.xticks(rotation=45)
        st.pyplot(fig)

with tab3:
    start_date = pd.to_datetime(
        st.date_input(
            "Start Date",
            value=df["fromDate"].min().date(),
            min_value=df["fromDate"].min().date(),
        )
    )
    end_date = pd.to_datetime(
        st.date_input(
            "End Date",
            value=df["fromDate"].max().date(),
            max_value=df["fromDate"].max().date(),
        )
    )
    df_filtered = df[(df["fromDate"] >= start_date) & (df["fromDate"] <= end_date)]
    period = st.selectbox("Select Period", [15, 30, 60, 90, 180])
    df_filtered = df_filtered.sort_values(by="fromDate")

    with st.expander("Time Asleep"):
        fig, ax = plt.subplots(figsize=(18, 6))
        ax.plot(
            df_filtered["fromDate"],
            df_filtered[f"asleepRoll{period}"],
            label="Asleep",
            color="red",
        )

        min_val = df_filtered[f"asleepRoll{period}"].min()
        max_val = df_filtered[f"asleepRoll{period}"].max()

        ax.axhline(max_val, color="g", linestyle="--")
        ax.axhline(min_val, color="b", linestyle="--")

        # Find dates for min and max values
        min_date = df_filtered.loc[
            df_filtered[f"asleepRoll{period}"] == min_val, "fromDate"
        ].iloc[0]
        max_date = df_filtered.loc[
            df_filtered[f"asleepRoll{period}"] == max_val, "fromDate"
        ].iloc[0]

        # Add text annotations
        ax.annotate(
            f"Min: {round(min_val,2)}",
            xy=(min_date, min_val),
            xytext=(10, 10),
            textcoords="offset points",
            color="b",
        )

        ax.annotate(
            f"Max: {round(max_val,2)}",
            xy=(max_date, max_val),
            xytext=(10, -10),
            textcoords="offset points",
            color="g",
        )

        ax.set_title(f"{period}-day Rolling Average Sleep")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sleep")
        plt.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with st.expander("Sleep BPM"):
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(df_filtered["fromDate"], df_filtered[f"sleepBPMRoll{period}"])
        ax.set_title(f"{period}-day Rolling Average Sleep BPM")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sleep BPM")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with st.expander("Efficiency"):
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(df_filtered["fromDate"], df_filtered[f"efficiencyRoll{period}"])
        # ax.axhline(df_filtered[f'sleepBPMRoll{period}'].max(), color='g', linestyle='--')
        # ax.axhline(df_filtered[f'sleepBPMRoll{period}'].min(), color='b', linestyle='--')

        # min_date = df_filtered.loc[df_filtered[f'sleepBPMRoll{period}'].idxmin()]['fromDate']
        # ax.text(min_date, (df_filtered[f'sleepBPMRoll{period}'].min()*1.05), f'Min: {round(df_filtered[f"sleepBPMRoll{period}"].min(),2)}',
        #         color='b', ha='right', va='center')
        # ax.text(min_date, (df_filtered[f'sleepBPMRoll{period}'].max()*0.95), f'Max: {round(df_filtered[f"sleepBPMRoll{period}"].max(),2)}',
        #         color='g', ha='right', va='center')

        ax.set_title(f"{period}-day Rolling Average Efficiency")
        ax.set_xlabel("Date")
        ax.set_ylabel("Efficiency")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with st.expander("Sleep HRV"):
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(df_filtered["fromDate"], df_filtered[f"sleepHRVRoll{period}"])
        ax.set_title(f"{period}-day Rolling Average Sleep HRV")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sleep HRV")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with st.expander("Raw Data"):
        st.dataframe(df_filtered)

with tab4:
    start_date_box = pd.to_datetime(
        st.date_input("Start Date for Boxplot", value=df["fromDate"].min().date())
    )
    end_date_box = pd.to_datetime(
        st.date_input("End Date for Boxplot", value=df["fromDate"].max().date())
    )
    st.text(
        "Top and Bottom 1% of outliers are removed from each metric before plotting"
    )
    df_box = df[(df["fromDate"] >= start_date_box) & (df["fromDate"] <= end_date_box)]
    df_box = df_box[["asleep", "quality", "deep", "sleepBPM", "sleepHRV", "efficiency"]]
    for col in df_box.columns:
        lower_bound = df_box[col].quantile(0.01)
        upper_bound = df_box[col].quantile(0.99)
        df_box = df_box[(df_box[col] > lower_bound) & (df_box[col] < upper_bound)]

    # Boxplots
    # Now create boxplots
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 16))
    plt.suptitle("Boxplots", size=20)
    for i, col in enumerate(df_box.columns):
        sns.boxplot(x=col, data=df_box, ax=axes[i // 2, i % 2])
        axes[i // 2, i % 2].set_title(f"{col}")
    plt.tight_layout()
    st.pyplot(fig)

with tab5:
    start_date_hist = pd.to_datetime(
        st.date_input(
            "Start Date",
            value=df["fromDate"].min().date(),
            min_value=df["fromDate"].min().date(),
            key=hash("start_date_hist"),
        )
    )
    end_date_hist = pd.to_datetime(
        st.date_input(
            "End Date",
            value=df["fromDate"].max().date(),
            max_value=df["fromDate"].max().date(),
            key=hash("end_date_hist"),
        )
    )
    df_hist = df[
        (df["fromDate"] >= start_date_hist) & (df["fromDate"] <= end_date_hist)
    ]

    with st.expander("Sleep Time"):
        df_slp_hist = df_hist[(df_hist["asleep"] >= 3) & (df_hist["asleep"] <= 12)]
        fig, ax = plt.subplots(figsize=(16, 6))
        n, bins, patches = ax.hist(
            df_slp_hist["asleep"], bins=(12 - 3) * 2, edgecolor="black", density=True
        )

        # Add bell curve
        mu, std = stats.norm.fit(df_slp_hist["asleep"])
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p, "k", linewidth=2)

        ax.set_title("Histogram of Asleep Time with Normal Distribution")
        ax.set_xlabel("Asleep Time (hours)")
        ax.set_ylabel("Density")
        st.pyplot(fig)

    with st.expander("Sleep BPM"):
        fig, ax = plt.subplots(figsize=(16, 6))
        df_hist = df_hist.dropna(subset=["sleepBPM"])
        if not df_hist.empty:
            n, bins, patches = ax.hist(
                df_hist["sleepBPM"],
                bins=int((90 - 40) / 2.5),
                edgecolor="black",
                range=(40, 90),
                density=True,
            )

            # Add bell curve
            mu, std = stats.norm.fit(df_hist["sleepBPM"])
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = stats.norm.pdf(x, mu, std)
            ax.plot(x, p, "k", linewidth=2)

            ax.set_title("Histogram of Sleep BPM with Normal Distribution")
            ax.set_xlabel("Sleep BPM")
            ax.set_ylabel("Density")
            st.pyplot(fig)
        else:
            st.warning("No data available for Sleep BPM")

    with st.expander("Sleep HRV"):
        fig, ax = plt.subplots(figsize=(16, 6))
        df_hist = df_hist.dropna(subset=["sleepHRV"])
        if not df_hist.empty:
            n, bins, patches = ax.hist(
                df_hist["sleepHRV"],
                bins=26,
                edgecolor="black",
                range=(20, 150),
                density=True,
            )

            # Add bell curve
            mu, std = stats.norm.fit(df_hist["sleepHRV"])
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = stats.norm.pdf(x, mu, std)
            ax.plot(x, p, "k", linewidth=2)

            ax.set_title("Histogram of Sleep HRV with Normal Distribution")
            ax.set_xlabel("Sleep HRV")
            ax.set_ylabel("Density")
            st.pyplot(fig)
        else:
            st.warning("No data available for Sleep HRV")

with tab6:
    # Exclude columns that start with 'Roll' or contain a '/'
    dfSimple = df.drop(
        columns=[col for col in df.columns if "Roll" in col or "/" in col], axis=1
    )
    correlation_matrix = round(dfSimple.corr(numeric_only=True), 4)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)
    st.subheader("Top 10 Correlated Values w/ Sleep Time")
    top_10_correlations = (
        abs(correlation_matrix["asleep"]).sort_values(ascending=False).head(10)
    )
    st.table(top_10_correlations)
