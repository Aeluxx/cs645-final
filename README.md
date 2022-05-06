# CS 645 Final Project: PaSQL
## By Jahnavi Tirunagari, Matthew Gregory, and Shruti Jasoria

## Project Description
We implement algorithms found in PaQL, analyzing and solving package queries to find the most optimal return value.

## How To Use
There are two ways to use this program, each with Python

### Method 1: Default
Running the program with no arguments (through an IDE or command line) will use the default values in the main.py main method. To edit these values, you can change the four fields 'data_sizes', 'cluster_sizes', 'to_test', and 'mode'.

#### data_sizes
A tuple of (minimum data size, maximum data size, data increment). This means that for a value of (0.1, 1, 0.1), the code will try all data sizes of 0.1, 0.2, ..., 1.

#### cluster_sizes
A tuple of (minimum cluster size, maximum cluster size, dclusterata increment). This means that for a value of (10, 100, 10), the code will try all data sizes of 10, 20, ..., 100.

#### to_test
An array of all queries you want to test, which will be run through in sequence.

#### mode
Which solver you want to use to make the package query. The values are as such: 0 = KMeans, 1 = KDTree, 2 = KMeans (min), 3 = Gaussian, 4 = Gaussian (min), 5 = QuadTree, 6+ = Direct.

### Method 2: Arguments
You can run the program with exactly 5 arguments to make a specific query. These arguments are as such:

#### data_size
The ratio of data to use in the analysis

#### cluster_size
The cluster size input for the model

#### query
Which query to use. Queries are defined in queries.py, and can be edited as per the following:

flag = True (Maximize) or False (Minimize), A0 = Column to maximize or minimize, None if count, constraints = List of [column name, lower bound, upper bound], count_constraint = {LC: Lower Bound on # tuples selected, UC: Upper Bound on # tuples selected}. Queries 1-4 are the 4 queries given for the project. Queries 5 and 6 are the simple queries we ran on the small dataset in our report.

#### mode
Same as in Method 1, defines the mode of solver to use in the package query.

#### file
A .csv file to run the query on. Starts in the same directory as the python file. For example, if tpch.csv is in this file's directory, the argument would simply be tpch.csv.

## Example Use
As an example, to run 100% of the data used (ratio 1) and 10 clusters with Query #5 (Small Query 1) on the data in tpch_small.csv with the KMeans algorithm (id = 0), you would run the command:

`python3 main.py 1 10 5 0 tpch_small.csv`

## Requirements
To use this software, you must have IBM's CPLEX installed. If you are using Windows, then the code should work as-is. If you are using Mac OS, you must edit the code in main.py to use the path to your CPLEX install location - for us, this was defined as `ilp = pulp.CPLEX_CMD(path="/Applications/CPLEX_Studio221/cplex/bin/x86-64_osx/cplex")`.

