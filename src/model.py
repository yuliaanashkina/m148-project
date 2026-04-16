import polars as pl

test_journeys = pl.read_csv("data/open_journeys1_flattened_all0.csv")
print(test_journeys.columns)
print(test_journeys.schema)
print(test_journeys.head())