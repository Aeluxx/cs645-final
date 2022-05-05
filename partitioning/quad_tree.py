import math
from numpy import partition

import pandas as pd


PARTITIONS = []

def genQuadTree(df : pd.DataFrame, stopping_count : int):
	df_size = len(df.index)
	## Empty partition. No need to move forward
	if df_size == 0:
		return
	## If the df has instances less than the stopping count, add it to the list
	## of partitions. No need to recurse anymore
	if df_size <= stopping_count:
		PARTITIONS.append(df)
		return
	## Split dataframes 
	splits = []
	attributes = df.columns
	for attribute in attributes:
		if attribute == 'id':
			continue
		## First split happens through the input df
		if len(splits) == 0:
			median = df[attribute].median()
			split1 = df[df[attribute] <= median]
			split2 = df[df[attribute] > median]
			splits.append(split1)
			splits.append(split2)
		else:
		## Subsequent splits are performed on the already existing splits	
			new_splits = []
			for split in splits:
				median = split[attribute].median()
				split1 = split[split[attribute] <= median]
				split2 = split[split[attribute] > median]
				new_splits.append(split1)
				new_splits.append(split2)
			splits = new_splits
	## recursion time		
	for split in splits:
		genQuadTree(split, stopping_count)	


def genRepresenation(df, stopping_count):
	print("Generating partitions")
	genQuadTree(df, stopping_count)
	num_partitions = len(PARTITIONS)
	for label in range(num_partitions):
		partition = PARTITIONS[label]
		partition["cluster_label"] = label
		filepath = "part" + str(label) + ".csv"
		partition.to_csv(path_or_buf=filepath, index=False)
	final_partition = pd.concat(PARTITIONS)
	rep_df = final_partition.groupby("cluster_label", as_index=False).mean()
	rep_df.drop("id", axis=1, inplace=True)
	return rep_df


if __name__ == "__main__":
	df = pd.read_csv('../tpch.csv', sep=',')
	df_size = math.ceil(len(df) * 0.1)
	df = df.head(df_size)
	print("Data frame size: ", df_size)
	stop_size = math.ceil(df_size * 0.1)
	rep_df = genRepresenation(df, stop_size)
	print(rep_df.head())
	print("Total number of clusters: ", len(rep_df.index))

	## Comment out the following for sanity check
	# num_partions = len(PARTITIONS)
	# print("Partions generated: ", num_partions)
	# clusters_total_instances = 0
	# for i in range(num_partions):
	# 	cluster = PARTITIONS[i]
	# 	cluster_size = len(cluster.index)
	# 	# print("Size of cluster ", i, ": ", cluster_size)
	# 	clusters_total_instances += cluster_size
	# print("Total Instances: ", clusters_total_instances)	
