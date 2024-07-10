```cpp
double adjacency_matrix[GRAPH_ORDER][GRAPH_ORDER];
double max_diff = 0.0;
double min_diff = 1.0;
double total_diff = 0.0;

double pagerank[GRAPH_ORDER];
double initial_rank = 1.0 / GRAPH_ORDER;

double damping_value = (1.0 - DAMPING_FACTOR) / GRAPH_ORDER;
double diff = 1.0;
size_t iteration = 0;
double sstart = omp_get_wtime();
double elapsed = omp_get_wtime() - sstart;
double time_per_iteration = 0;
double new_pagerank[GRAPH_ORDER];

double pagerank_total = 0.0;
for (int i = 0; i < GRAPH_ORDER; i++) {
    pagerank_total += pagerank[i];
}

```

# Loops
```c
pagerank: READ
adjacency_matrix: READ
new_pagerank: WRITE, reduction on pagerank[i]
outdegree: READ/WRITE, reduction on write


for (int i = 0; i < GRAPH_ORDER; i++) {
    for (int j = 0; j < GRAPH_ORDER; j++) {
    if (adjacency_matrix[j][i] == 1.0) {
        int outdegree = 0;

        for (int k = 0; k < GRAPH_ORDER; k++) {
            if (adjacency_matrix[j][k] == 1.0) {
                outdegree++;
            }
        }
        new_pagerank[i] += pagerank[j] / (double)outdegree;
    }
    }
}
```