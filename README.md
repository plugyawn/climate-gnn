# CLIMNET: Graph Neural Network for Climate data.

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)

<div align = center>
<a href = "github.com/plugyawn/gossip"><img width="600px" height="600px" src= "https://github.com/plugyawn/climate-gnn/assets/76529011/b6a55787-b720-44ef-b21e-45f1bc2aba26"></a>
</div>

-------------------

## Data Processing


Please refer to `temporal_gnn.ipynb` for the most up-to-date information about this repository.

This code snippet demonstrates the processing of data to create a list of network graphs. The graphs are constructed from adjacency matrices and coordinate information for a range of years.

### Steps

1. **Loading Data**: The code starts by iterating over the range of years from 2003 to 2015 using the `range` function. For each year, it loads an adjacency matrix from a file specified by the `ADJ_PATH` format string and coordinate information from a CSV file specified by the `COORD_PATH` format string.

2. **Preprocessing**: The code performs several preprocessing steps on the adjacency matrix:
   - The adjacency matrix is made symmetric by adding its transpose to itself using `adj_matrix = adj_matrix + adj_matrix.transpose()`.
   - Some rows and columns are removed from the adjacency matrix using slicing: `adj_matrix = adj_matrix[:4523, :4523]`.
   - Values in the adjacency matrix below a threshold (0.7) are set to zero: `adj_matrix[adj_matrix  < 0.7] = 0`.

3. **Graph Construction**: After preprocessing the adjacency matrix, a graph `G` is created using the `nx.from_numpy_matrix` function, which converts the adjacency matrix into a networkx graph.

4. **Node Attribute Setting**: The code creates a dictionary `lat_lon_dict` to store the latitude, longitude, and rain information for each node. It iterates over the coordinate information dataframe and populates the dictionary with the corresponding values.

5. **Assigning Node Attributes**: The node attributes are assigned to the graph `G` using the `nx.set_node_attributes` function. The `pos` attribute is set using the `lat_lon_dict`.

6. **Filtering Central Nodes**: The code identifies the central nodes within a specific geographical region. It iterates over the nodes of the graph `G` and checks if the latitude and longitude values fall within the defined range. The central nodes are added to the `central_nodes` list.

7. **Subgraph Extraction**: The graph `G` is filtered to create a subgraph containing only the central nodes using `G.subgraph(central_nodes)`.

8. **Appending to Graph List**: The subgraph `G` is added to the `graph_list`.

9. Filter the dataframe to include only rows corresponding to central nodes.

10. Initialize empty lists to store processed features and weights.

11. Iterate over the `graph_list` (except the last element) and performs the following sub-steps:
   - Converts the adjacency matrix of the graph to a numpy matrix.
   - Reads coordinate information from the `COORD_PATH` file corresponding to the year.
   - Extracts node features by converting the graph's node attributes (latitude, longitude, and rain) to a tensor.
   - Constructs the edge list by identifying edges with non-zero weights in the adjacency matrix.
   - Calculates edge weights based on the adjacency matrix values.
   - Prepares a central position dictionary for the nodes.
   - Appends the processed features and weights to the respective lists.

12. Stores the processed features and weights in various batch variables.
