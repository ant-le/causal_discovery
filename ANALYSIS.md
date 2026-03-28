## General Guidelines
- Genrally prefer error plots with mean over barplots when reporting mean 
values with deviation
- In tables, highlight per column best values in bold
- Ensure consistent colours for each model and make sure to viusally seperate 
amortised models from explicit models  
(explicit models in grey tones as the focus is on amortised models)
- make sure to place the tables such that it goes with the flow of reading
- make sure that for each section you only include test sets that are 
relevant for the given section (see dg_2pretrain_multimodel.yml)
- Remove the DG-prefix
- Remove all subsections and just start the respective 
paragraph with the topic in bold

## Change the results section in the following way:

5.1 Generalisation

- Make the xaxis Labels shorter by ommiting `d` and `n` and
by not using ID OOD prefix but highlighting ID vs OOD visually
- dont use the green line in all graphs

5.2 Transfer

- In Compound Shift , I want a graph which uses the normalised E-SID.
Otherwise comparison fails. I would divide plot in 1) ID, 2) Dimensions, 
3) NumSamples in one row and choose the fitting datasets.

5.3 Uncertaintly

- Remove the labels from the Scatterplot and make the layout tight
and the figures bigger


## OVERALL
Except the mentionned plots and tables, please go over each
plot and graph in analysis and the thesis generated folder
and for each one evaluate its purpuse and if its actually 
needed. I want for eaach resaerch question a map of all 
researches with importance and description.
Afterwards, draft potential additions and add them
to the table
