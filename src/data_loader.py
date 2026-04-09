import polars as pl

DATA_PATH = "data/dat_train1.csv"

# lazy load
df = pl.scan_csv(DATA_PATH)


## task 1

#1.1
num_rows = df.select(pl.len()).collect().item()
#1.2
num_unique_ids = df.select(pl.col("id").n_unique()).collect().item()

#1.3
earliest, latest = df.select([
    pl.col("event_timestamp").min().alias("earliest"),
    pl.col("event_timestamp").max().alias("latest")
]).collect().row(0)



print("Task 1.1 - Number of rows:", num_rows)

print("Task 1.2 - Number of unique IDs:", num_unique_ids)

print("Task 1.3 - Earliest timestamp:", earliest)
print("Task 1.3 - Latest timestamp:", latest)
## task 2

#2.1
total = num_rows

unique = df.unique(
    subset=["id", "event_name", "event_timestamp"]
).select(pl.len()).collect().item()

duplicates = total - unique
proportion = duplicates / total

# remove duplicates
df_clean = df.unique(subset=["id", "event_name", "event_timestamp"])

# fix the action counter
df_clean = df_clean.sort(["id", "event_timestamp"]).with_columns(
    pl.int_range(1, pl.len() + 1).over("id").alias("journey_steps_until_end")
)

df_clean.collect().write_csv("data/dat_train1_clean.csv")

#2.2
df_clean = pl.scan_csv("data/dat_train1_clean.csv")

num_rows_clean = df_clean.select(pl.len()).collect().item()

print("Task 2.1 - Number of duplicates:", duplicates)
print("Task 2.1 - Proportion of duplicates:", proportion)

print("Task 2.2 - Rows after removing duplicates:", num_rows_clean)


## task 3

# sample
unique_ids = df.select("id").unique().collect()
# sample 100k ids
sample_ids = unique_ids.sample(n=100000, shuffle=True)
# filter full df by sampled ids
df_sample = df.filter(pl.col("id").is_in(sample_ids["id"])).collect()

# parse timestamp once
df_sample = df_sample.with_columns(
    pl.col("event_timestamp").str.strptime(
        pl.Datetime,
        format="%Y-%m-%dT%H:%M:%SZ"
    )
)

# # 3.1
actions_per_journey = df_sample.group_by("id").agg(
    pl.len().alias("num_actions")
)

journey_time = df_sample.group_by("id").agg(
    (pl.col("event_timestamp").max() - pl.col("event_timestamp").min()).alias("duration")
)
journey_time_seconds = journey_time.with_columns(
    pl.col("duration").dt.total_seconds().alias("duration_sec")
)

# 3.2
common_actions = df_sample.group_by("event_name").agg(
    pl.len().alias("count")
).sort("count", descending=True)

# 3.3
time_between = df_sample.sort(["id", "event_timestamp"]).with_columns(
    (pl.col("event_timestamp") - pl.col("event_timestamp").shift(1))
    .over("id")
    .alias("time_diff")
)
time_between_seconds = time_between.with_columns(
    pl.col("time_diff").dt.total_seconds().alias("time_diff_sec")
)

# # 3.4
# print(actions_per_journey.describe())
# actions_per_journey.to_pandas().hist(column="num_actions")

# figures
import matplotlib.pyplot as plt


print("Task 3.1 - Journey length stats:")
print(actions_per_journey.select([
    pl.col("num_actions").mean().alias("mean"),
    pl.col("num_actions").median().alias("median"),
    pl.col("num_actions").std().alias("std"),
    pl.col("num_actions").min().alias("min"),
    pl.col("num_actions").max().alias("max"),
]))

print("Task 3.1 - Journey duration (seconds):")
print(journey_time_seconds.select([
    pl.col("duration_sec").mean().alias("mean_sec"),
    pl.col("duration_sec").median().alias("median_sec"),
    pl.col("duration_sec").max().alias("max_sec"),
]))

print("Task 3.2 - Most common actions:")
print(common_actions.head(10))
print("Task 3.3")
print(time_between_seconds.select([
    pl.col("time_diff_sec").mean().alias("mean_sec"),
    pl.col("time_diff_sec").median().alias("median_sec"),
    pl.col("time_diff_sec").max().alias("max_sec"),
]))

print("Task 3.4 - Distribution of journey lengths:")
print(actions_per_journey.describe())

print("Figure 1: Journey length distribution")
actions_per_journey.to_pandas().hist(column="num_actions", bins=50)
plt.xscale("log")
plt.title("Journey Length Distribution (log scale)")
plt.xlabel("Number of Actions")
plt.ylabel("Frequency")
plt.savefig("fig1_journey_length.png")
plt.show()
plt.clf()

print("Figure 2: Time between actions distribution")
time_between_seconds.select("time_diff_sec").to_pandas().hist(bins=50)
plt.xscale("log")
plt.title("Time Between Actions (log scale in seconds)")
plt.xlabel("Seconds")
plt.ylabel("Frequency")
plt.savefig("fig2_time_between.png")
plt.show()
plt.clf()

print("Figure 3: Most common actions")
common_actions.head(10).to_pandas().plot.bar(
    x="event_name", y="count"
)
plt.title("Top 10 Most Common Actions")
plt.xlabel("Action")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig("fig3_common_actions.png")
plt.show()
plt.clf()


## task 4
def flatten_journeys_parquet(input_csv_path, output_parquet_path):
    """
    Removes duplicates and flattens an event csv file so each journey (id)
    is one row, with the ordered journey stored as structs
    ["event_timestamp", "ed_id"] plus summary features.
    """
    q = pl.scan_csv(input_csv_path)
    q = q.unique(subset=["id", "ed_id", "event_timestamp"])
    q = q.with_columns(
        pl.col("event_timestamp").str.to_datetime(time_zone="UTC")
    )
    q = q.sort(["id", "event_timestamp", "ed_id"])

    df = (
        q.group_by("id")
        .agg([
            pl.struct(["event_timestamp", "ed_id"])
            .alias("journey"),

            pl.len().alias("num_actions"),
            pl.col("ed_id").n_unique().alias("num_unique_actions"),

            pl.col("event_timestamp").min().alias("start_time"),
            pl.col("event_timestamp").max().alias("end_time"),

            (pl.col("event_timestamp").max() - pl.col("event_timestamp").min())
            .dt.total_seconds()
            .alias("duration_seconds"),

            pl.col("ed_id").first().alias("first_action"),
            pl.col("ed_id").last().alias("last_action"),
        ])
    )

    df.sink_parquet(output_parquet_path)

training_csv_path = "data/dat_train1.csv"
output_path = "data/journeys_flattened.parquet"

flatten_journeys_parquet(training_csv_path, output_path)

df = pl.read_parquet(output_path)
print("Task 4 - Flattened journey dataset preview:")
print(df.head())
