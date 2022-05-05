import math

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

CLUSTER_COLS = ["sum_base_price", "sum_disc_price", "sum_charge", "avg_qty",
				"avg_price", "avg_disc", "sum_qty",  "count_order",  "p_size",  
				"ps_min_supplycost", "revenue", "o_totalprice", "o_shippriority"]

def generateParitions(df):
	normalized_df=(df-df.min())/(df.max()-df.min())
	normalized_df["id"] = df["id"]
	print("Starting partitioning")
	# Using manhattan distance as it is supposed to perform well for high 
	# dimensional data
	dbscan = DBSCAN(metric='manhattan').fit(normalized_df[CLUSTER_COLS])
	## TODO - remove these print statements later
	print("Number of cluster indices: ", len(dbscan.core_sample_indices_))
	print("Total labels: ", len((dbscan.labels_)))
	unique_labels = set(dbscan.labels_)
	print("Unique labels: ", unique_labels)
	noise = np.count_nonzero(dbscan.labels_ == -1)
	print("Noise: ", noise)
	tagged = len(df.index) - noise
	print("Tagged: ", tagged)
	print("Number of components: ", len(dbscan.components_))
	### Add the cluster_label column
	df["cluster_label"] = dbscan.labels_
	## Filter out instances tagged as noise
	df = df[df["cluster_label"]  != -1]
	for label in unique_labels:
		if label == -1:
			continue
		filepath = "part" + str(label) + ".csv"
		cluster = df[df["cluster_label"]  == label]
		cluster.to_csv(path_or_buf=filepath, index=False)
	rep_df = df.groupby("cluster_label", as_index=False).mean()
	rep_df.drop("id", axis=1, inplace=True)
	print(rep_df.head())
	return rep_df
	

if __name__ == "__main__":
	df = pd.read_csv('../tpch.csv', sep=',')
	size = math.ceil(len(df) * .01)
	df = df.head(size)
	print("Number rows in df: ", len(df.index))
	rep_df = generateParitions(df)
