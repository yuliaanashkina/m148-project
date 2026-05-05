"""One-shot data engineering pipeline (tasks 1-7).

This file runs the full data prep workflow as side effects at module
top-level: it reads ``data/dat_train1.csv``, writes cleaned CSVs and
parquets back into ``data/``, and renders figures into ``figures/``.
Because every line below executes on import, this is a SCRIPT, not a
module. Importing it from another file (or accidentally via Jupyter)
will trigger the whole pipeline and crash if the raw data is absent.

Run with ``python -m src.data_engineering.data_loader`` instead.
"""

from pathlib import Path

import polars as pl

if __name__ != "__main__":
    raise ImportError(
        "src.data_engineering.data_loader is a script with import-time "
        "side effects. Run it directly (python -m "
        "src.data_engineering.data_loader) instead of importing it."
    )

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SUBMISSION_DIR = RESULTS_DIR / "submissions"
FIGURE_DIR = PROJECT_ROOT / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "dat_train1.csv"

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

df_clean.collect().write_csv(DATA_DIR / "dat_train1_clean.csv")

#2.2
df_clean = pl.scan_csv(DATA_DIR / "dat_train1_clean.csv")

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
plt.savefig(FIGURE_DIR / "fig1_journey_length.png")
plt.show()
plt.clf()

print("Figure 2: Time between actions distribution")
time_between_seconds.select("time_diff_sec").to_pandas().hist(bins=50)
plt.xscale("log")
plt.title("Time Between Actions (log scale in seconds)")
plt.xlabel("Seconds")
plt.ylabel("Frequency")
plt.savefig(FIGURE_DIR / "fig2_time_between.png")
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
plt.savefig(FIGURE_DIR / "fig3_common_actions.png")
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

training_csv_path = DATA_DIR / "dat_train1.csv"
output_path = DATA_DIR / "journeys_flattened.parquet"

flatten_journeys_parquet(training_csv_path, output_path)

df = pl.read_parquet(output_path)
print("Task 4 - Flattened journey dataset preview:")
print(df.head())

## task 5
# Compare successful journeys vs incomplete journeys
# Assumption:
# - one id = one full journey
# - successful = last action is "order shipped"
# - incomplete = NOT successful and then 60+ days of inactivity after the last event

## task 5
# Compare successful journeys vs incomplete journeys
# Assumption:
# - one id = one full journey
# - successful = last action is "order shipped"
# - incomplete = NOT successful and then 60+ days of inactivity after the last event

ORDER_SHIPPED_ID = 999  # TODO: replace with the actual ed_id for "order shipped"

# read flattened journeys from task 4
journeys = pl.read_parquet(DATA_DIR / "journeys_flattened.parquet")

# get the latest timestamp in the full raw dataset
dataset_end_time = (
    pl.scan_csv(DATA_PATH)
    .with_columns(
        pl.col("event_timestamp").str.to_datetime(time_zone="UTC")
    )
    .select(pl.col("event_timestamp").max().alias("dataset_end_time"))
    .collect()
    .item()
)

## task 5
# Compare successful journeys vs incomplete journeys
# Assumption:
# - one id = one full journey
# - successful = last action is "order shipped"
# - incomplete = NOT successful and then 60+ days of inactivity after the last event

import matplotlib.pyplot as plt
import pandas as pd

ORDER_SHIPPED_ID = 28  # replace if needed

# read flattened journeys from task 4
journeys = pl.read_parquet(DATA_DIR / "journeys_flattened.parquet")

# get the latest timestamp in the full raw dataset
dataset_end_time = (
    pl.scan_csv(DATA_PATH)
    .with_columns(
        pl.col("event_timestamp").str.to_datetime(time_zone="UTC")
    )
    .select(pl.col("event_timestamp").max().alias("dataset_end_time"))
    .collect()
    .item()
)

# label journeys
journeys_labeled = journeys.with_columns([
    (pl.col("last_action") == ORDER_SHIPPED_ID).alias("is_successful"),
    (
        (pl.lit(dataset_end_time) - pl.col("end_time")).dt.total_days() >= 60
    ).alias("inactive_60_days"),
]).with_columns([
    (
        pl.when(pl.col("is_successful"))
        .then(pl.lit("successful"))
        .when((~pl.col("is_successful")) & pl.col("inactive_60_days"))
        .then(pl.lit("incomplete"))
        .otherwise(pl.lit("other"))
    ).alias("journey_status")
])

print("\nTask 5.1 - Journey status counts:")
status_counts = (
    journeys_labeled.group_by("journey_status")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
)
print(status_counts)

# Visual 1: journey status counts
status_counts.to_pandas().plot.bar(x="journey_status", y="count", legend=False)
plt.title("Journey Status Counts")
plt.xlabel("Journey Status")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "fig4_journey_status_counts.png")
plt.show()
plt.clf()

# keep only the two groups requested
comparison_df = journeys_labeled.filter(
    pl.col("journey_status").is_in(["successful", "incomplete"])
)

# overall summary stats by group
summary_stats = (
    comparison_df.group_by("journey_status")
    .agg([
        pl.len().alias("n_journeys"),

        pl.col("num_actions").mean().alias("avg_num_actions"),
        pl.col("num_actions").median().alias("median_num_actions"),
        pl.col("num_actions").std().alias("std_num_actions"),

        pl.col("num_unique_actions").mean().alias("avg_num_unique_actions"),
        pl.col("num_unique_actions").median().alias("median_num_unique_actions"),

        pl.col("duration_seconds").mean().alias("avg_duration_seconds"),
        pl.col("duration_seconds").median().alias("median_duration_seconds"),
        pl.col("duration_seconds").max().alias("max_duration_seconds"),
    ])
    .sort("journey_status")
)

print("\nTask 5.2 - Summary stats: successful vs incomplete")
print(summary_stats)

comparison_pd = comparison_df.to_pandas()

# Visual 2: boxplot of number of actions
comparison_pd.boxplot(column="num_actions", by="journey_status")
plt.title("Number of Actions by Journey Type")
plt.suptitle("")
plt.xlabel("Journey Type")
plt.ylabel("Number of Actions")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "fig5_num_actions_boxplot.png")
plt.show()
plt.clf()

# Visual 3: boxplot of journey duration
comparison_pd.boxplot(column="duration_seconds", by="journey_status")
plt.title("Journey Duration by Journey Type")
plt.suptitle("")
plt.xlabel("Journey Type")
plt.ylabel("Duration (seconds)")
plt.yscale("log")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "fig6_duration_boxplot.png")
plt.show()
plt.clf()

# Visual 4: histogram of number of actions
for status in ["successful", "incomplete"]:
    subset = comparison_pd[comparison_pd["journey_status"] == status]
    plt.hist(subset["num_actions"], bins=50)
    plt.title(f"Distribution of Number of Actions - {status.capitalize()}")
    plt.xlabel("Number of Actions")
    plt.ylabel("Frequency")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"fig7_num_actions_hist_{status}.png")
    plt.show()
    plt.clf()

# Visual 5: histogram of duration
for status in ["successful", "incomplete"]:
    subset = comparison_pd[comparison_pd["journey_status"] == status]
    plt.hist(subset["duration_seconds"], bins=50)
    plt.title(f"Distribution of Journey Duration - {status.capitalize()}")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Frequency")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"fig8_duration_hist_{status}.png")
    plt.show()
    plt.clf()

# first action distribution by group
first_action_summary = (
    comparison_df.group_by(["journey_status", "first_action"])
    .agg(pl.len().alias("count"))
    .sort(["journey_status", "count"], descending=[False, True])
)

print("\nTask 5.3 - Most common first actions by group")
print(
    first_action_summary
    .group_by("journey_status")
    .head(10)
)

# Visual 6: top first actions by group
for status in ["successful", "incomplete"]:
    subset = (
        first_action_summary
        .filter(pl.col("journey_status") == status)
        .head(10)
        .to_pandas()
    )
    subset.plot.bar(x="first_action", y="count", legend=False)
    plt.title(f"Top 10 First Actions - {status.capitalize()}")
    plt.xlabel("First Action")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"fig9_first_actions_{status}.png")
    plt.show()
    plt.clf()

# last action distribution by group
last_action_summary = (
    comparison_df.group_by(["journey_status", "last_action"])
    .agg(pl.len().alias("count"))
    .sort(["journey_status", "count"], descending=[False, True])
)

print("\nTask 5.4 - Most common last actions by group")
print(
    last_action_summary
    .group_by("journey_status")
    .head(10)
)

# Visual 7: top last actions for incomplete journeys (drop-off points)
dropoff_actions = (
    last_action_summary
    .filter(pl.col("journey_status") == "incomplete")
    .head(10)
    .to_pandas()
)

dropoff_actions.plot.bar(x="last_action", y="count", legend=False)
plt.title("Top 10 Drop-off Actions for Incomplete Journeys")
plt.xlabel("Last Action")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "fig10_incomplete_dropoff_actions.png")
plt.show()
plt.clf()

# optional: save labeled dataset
journeys_labeled.write_parquet(DATA_DIR / "journeys_labeled.parquet")
print(f"\nSaved labeled journeys to {DATA_DIR / 'journeys_labeled.parquet'}")


## task 6
# Create training data for predictive modeling using Option A:
# one snapshot per journey

import math
import polars as pl

# We assume journeys_labeled already exists from Task 5 and contains:
# - id
# - journey (list of structs with event_timestamp and ed_id)
# - journey_status in {"successful", "incomplete", "other"}

# --------------------------------------------------
# 6.1 Keep only journeys with clear final outcomes
# --------------------------------------------------
model_base = journeys_labeled.filter(
    pl.col("journey_status").is_in(["successful", "incomplete"])
)

print("\nTask 6.1 - Number of labeled journeys kept for modeling:")
print(
    model_base.group_by("journey_status")
    .agg(pl.len().alias("count"))
    .sort("journey_status")
)

# --------------------------------------------------
# 6.2 Create one partial snapshot per journey
# --------------------------------------------------
# Rule:
# - Use the first 70% of actions
# - But keep at least 1 action
# - And never keep the full journey, so the model does not see the outcome step

def make_snapshot(row, frac=0.7):
    journey = row["journey"]
    n = len(journey)

    if n <= 1:
        k = 1
    else:
        k = max(1, min(n - 1, math.floor(frac * n)))

    prefix = journey[:k]

    timestamps = [step["event_timestamp"] for step in prefix]
    actions = [step["ed_id"] for step in prefix]

    start_time = timestamps[0]
    current_time = timestamps[-1]
    current_last_action = actions[-1]
    first_action = actions[0]

    prefix_duration_seconds = int((current_time - start_time).total_seconds()) if k > 1 else 0
    num_actions_so_far = k
    num_unique_actions_so_far = len(set(actions))

    if k > 1:
        avg_gap_seconds = prefix_duration_seconds / (k - 1)
        time_since_prev_action_seconds = int((timestamps[-1] - timestamps[-2]).total_seconds())
    else:
        avg_gap_seconds = 0
        time_since_prev_action_seconds = 0

    return {
        "id": row["id"],
        "label": 1 if row["journey_status"] == "successful" else 0,
        "final_outcome": row["journey_status"],

        # snapshot metadata
        "full_num_actions": n,
        "snapshot_num_actions": num_actions_so_far,
        "snapshot_frac_of_journey": num_actions_so_far / n,

        # prefix-based features only
        "first_action_so_far": first_action,
        "current_last_action": current_last_action,
        "num_unique_actions_so_far": num_unique_actions_so_far,
        "prefix_duration_seconds": prefix_duration_seconds,
        "avg_gap_seconds": avg_gap_seconds,
        "time_since_prev_action_seconds": time_since_prev_action_seconds,

        # optional raw prefix sequence
        "prefix_actions": actions,
    }

snapshot_rows = [make_snapshot(row) for row in model_base.iter_rows(named=True)]
training_df = pl.DataFrame(snapshot_rows)

print("\nTask 6.2 - Training dataset preview:")
print(training_df.head())

# --------------------------------------------------
# 6.3 Add count features for common actions
# --------------------------------------------------
# This turns sequence information into model-friendly numeric columns.

top_actions = (
    training_df.explode("prefix_actions")
    .group_by("prefix_actions")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
    .head(15)
    .get_column("prefix_actions")
    .to_list()
)

for action_id in top_actions:
    training_df = training_df.with_columns(
        pl.col("prefix_actions")
        .list.eval(pl.element() == action_id)
        .list.sum()
        .alias(f"action_count_{action_id}")
    )

print("\nTask 6.3 - Added action count features for top actions:")
print(top_actions)

# --------------------------------------------------
# 6.4 Save modeling dataset
# --------------------------------------------------
training_df.write_parquet(DATA_DIR / "journey_training_optionA.parquet")

# CSV cannot store nested list columns like prefix_actions
training_df_csv = training_df.drop("prefix_actions")
training_df_csv.write_csv(DATA_DIR / "journey_training_optionA.csv")

print("\nSaved training data to:")
print(DATA_DIR / "journey_training_optionA.parquet")
print(DATA_DIR / "journey_training_optionA.csv")

# --------------------------------------------------
# 6.5 Basic label balance check
# --------------------------------------------------
print("\nTask 6.5 - Label balance:")
print(
    training_df.group_by(["label", "final_outcome"])
    .agg(pl.len().alias("count"))
    .sort("label")
)

## task 7
# Fit a simple Random Forest model

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# --------------------------------------------------
# 7.1 Load training data
# --------------------------------------------------
df = pl.read_parquet(DATA_DIR / "journey_training_optionA.parquet")

# drop nested column + anything not usable
df_model = df.drop(["prefix_actions", "final_outcome"])

# convert to pandas for sklearn
df_pd = df_model.to_pandas()

print("\nTask 7.1 - Training data shape:")
print(df_pd.shape)

# --------------------------------------------------
# 7.2 Encode categorical variables
# --------------------------------------------------
# (these are numeric IDs but treated as categories)
cat_cols = ["first_action_so_far", "current_last_action"]

for col in cat_cols:
    le = LabelEncoder()
    df_pd[col] = le.fit_transform(df_pd[col])

# --------------------------------------------------
# 7.3 Define features + target
# --------------------------------------------------
X = df_pd.drop(columns=["label", "id"])
y = df_pd["label"]

# --------------------------------------------------
# 7.4 Fit Random Forest (with OOB score)
# --------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    oob_score=True,
    n_jobs=-1
)

rf.fit(X, y)

print("\nTask 7.2 - OOB Accuracy:")
print(rf.oob_score_)

# --------------------------------------------------
# 7.5 Feature importance
# --------------------------------------------------
importances = rf.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nTop features:")
print(importance_df.head(10))

# --------------------------------------------------
# 7.6 Plot feature importance
# --------------------------------------------------
top_k = 10
top_features = importance_df.head(top_k)

plt.barh(top_features["feature"], top_features["importance"])
plt.gca().invert_yaxis()
plt.title("Top Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "fig11_feature_importance.png")
plt.show()

## task 8
# Create a Kaggle-style submission file using the training data itself
# This is only a proof of concept because no separate test set is available.

import pandas as pd

# --------------------------------------------------
# 8.1 Use the same fitted model and feature matrix from Task 7
# --------------------------------------------------
# Assumes Task 7 already created:
# - rf
# - X
# - df_pd
#
# where:
# X = feature matrix
# df_pd["id"] = ids for each training row

train_probs = rf.predict_proba(X)[:, 1]

submission = pd.DataFrame({
    "id": df_pd["id"],
    "order_shipped": train_probs
})

# match requested format exactly
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
submission.to_csv(SUBMISSION_DIR / "kaggle_submission.csv", index=False)

print(f"\nSaved submission file to {SUBMISSION_DIR / 'kaggle_submission.csv'}")
print(submission.head())

# optional: quick summary of predicted probabilities
print("\nPrediction summary:")
print(submission["order_shipped"].describe())
