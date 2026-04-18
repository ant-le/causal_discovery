GENERAL:
- Use completely different colours for the error 
decomposition than for the models (e.g. grayscale)

START:
- result anchor does not make highlight the best
value per metric in bold (\textbf{}) and report 
thresholded DAG percentage instead of sample perc

RQ1:
- Adjust all USED Rq1 figures to 
span three rows where Dag validity
is the top row and the error decomposition is the
bottom row. Also remove the all-model references. Based on the 
error decomposition and DAG output also adjust the text
if needed
-> Basic Figures Done But 
    - now for each plot the shared XLabels Are gone!!! 
    - use tight_layout
    - and the headers should be above the top row
    and not above the moddle row
    - compound shift table seems off and the graph 
    headers are probably wrong as there is no 
    ID-task for SBM/WS/GRG, also Dag validity
    and error decomposition is not there for stress test

- For he Avici Dag validity vs Metrics there is no 
structure in the plot so please investiagte the 
avici metrics.json and see if there is any 
relation if conditioned: If so adjust the plot,
if not remove it


RQ2:
- IN `degradation_heatmap` make the speread estimate
not in bold and MUCH smaller

- Adjust the `rq2_node_transfer` and `RQ2SampleTransferFigure`
plot by also ordering these with: Dag Validity on
top row, then resilt row, last error decomposition
row. use the same Dag Validity logic here but plot
Avici AND DiBs

- In `rq2_worst_task_comparison`, label how the error
decomposition was scaled such taht the reader knows
and state it also in the description

- When deiscussing metrics sensitivity, remove the table
on `Family-level winner count` and ingest the data 
to see if there can be made any more inference 
on metrics 

RQ3:

- The comparison of NILL and sid, should the suggestive
line also be non-linear or should NIL scale 
linearly with sid?
- as a metric, maybe use MSE instead of Correlation
to account for non-leanrity?
- Remove table on spearman corelation as graph is 
enough in my opinion
