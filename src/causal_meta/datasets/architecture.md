# Goals

1. Fix two sets of Structural Causal Models (SCM)
> Here the basic idea is that meta-learning models want to model a 
posterior over SCMs (potentially over all possible SCMs), so there need
to be fixed i.d. and o.o.d. definitions for comparisons

- i.d. SCM family
- o.o.d. SCM families (multiple?, how can I group them?)
- Tipp: Distinguish family vs set:
    - Family: the generative process / parameter ranges.
    - Set: a finite sample of SCMs instantiated from that family for a particular 
    experiment run.

2. Calculate differences between i.d.d. SCMs and each o.o.d. SCM families
> Information Theoretic concpets should be used to quantify the difference
between SCM sets where both parts, graph structure and functional forms, 
should be compared. Also, new metrics for SCMs can be considered when 
applicable to Bayesian Infernece.

- o.o.d. meansure for each o.o.d. SCM family

# Components

1. Dataset Class containing all information stored (reusable to instantiate 
eval and train dataset with param, or wrapper)
    
    - Test sets need to be generated and stored in advance
    - Training samples can be generated 'on the fly' (sample set from family 
    -> sample from SCM famuily)
    - takes in parameters defining meta-data for o.o.d set, sizes, ...
    - Captures intenventions naturally
    - some general plotting
        - intervention
        - graph
        - i.d. vs. o.o.d.
    - some general descriptive statistics
        - metadata
        - o.o.d. distances
    - run  distance Functions quantifying the distance between sets of sets

2. Generator Functions to generate different sets of SCMs 
    - SCMFamily / Generator: can produce SCM instances given RNG.
    - SCM: knows its graph, mechanisms, noise, and can sample data.

3. Subfolders storing the test datasets, meta-data and results 
for a specific run for reprocubility

# Parameters 
> what are properties of an SCM which can be changed?

1. Required
- set of functional generators to include
- graphs properties (size, density)
- noise compositions
- share and sizes of o.o.d
- sample size
- Invertibility

2. Optional
- Train/Val Split
- Distance Functions chosen
- Intervention Sparsity
- Markov Equivalence Class (MEC) size
