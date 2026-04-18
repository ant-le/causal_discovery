Grahpics Results

1. All three-row tables comparing task families
(except node and sample count) must have the used 
task config on the xaxis, currently the xticks
are just numbers

2. please always use tight_layout

3. Sample and Node-Count Transfer currently have
two problems
- a very cluttered legend at the top, there only 
models should be
- make DiBs dag validty in oranges - same as D

4. The compound shift graph is currently missing 
data points (stress not having dag validity and
error decomposition)

5. The current Avici DAG plot doesnt show 
anything - maybe try if difference between
thresholded and samples leads to something,
if not remove the plot and tighten the 
paragraph


Appendix

1. Remove Section on metric definition
entirely as I have already discussed all in 
Mtehtodology
2. Remove Distance-Degradation Regression
entirely as it is not disucssed in the 
thesis
3. Reproducability - Many tables of
the hyperparameter alignment are still
N/A 
    - Fill the values from the config
    into the column instead of putting
    it below under Key Differences
    - Remove the Match column
    - Ours -> This Thesis
