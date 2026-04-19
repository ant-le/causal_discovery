NOW:
- Inconsistency in the description of DiBS. 
In the prose, DiBS is described as modeling nonlinear 
mechanisms with neural networks. But in Table 3.1, its 
mechanism class is listed as “Nonparametric (GP)”. 
Both cannot be the operative description at the same time 
without explanation. This needs correction. 
-> Look at avici paper to find correct notion

- There is also naming drift around the mechanism 
family used in training/evaluation:
the text refers to RFF,
a figure labels something like GPCDE,
appendices and comparison tables use GP or GP-CDE 
in some places.
-> Rename in Thesis and all grahps to RFF!!!

- checkpoint claim in appendix that best model is chosen based on F1 
-> change it to o.o.d mean sid and adjust the trainign config to reflect that

- in the first results table, also show inference time once and mention
it in one sentence

- Update the BCNP citation meta_learning (Dhir et al.) from an arXiv 
preprint to its official publication venue at ICLR 2025




FURTHER STUFF:
1. Test what seems AI generated
2. Re-Verify ALL claims and citations
3. Check plagiarism
4. Note in code and paper from where stuff was taken
(eval resembles both BayesDag and Avici)
5. Explain the difference in output between BCNP and AVICI 
in more detail in 3_Related_work. I would add an 
architectural graphic that shows (at least) both decoder 
architectures

- AVICI produces edge prob matrix -> then samples 
- BCNP produces DAG samples directly


BETTER EXPERIMENT:
1. Change NumNodes training
to 15-45 -> ood are 5/10 and 50/55/60
2. Potentially enhance stress testing if needed
to provide more of a ladder:
    
    - First: Compound Shift with less samples
    - Second: Compound Shift with more nodes
    - Third: Compound Shift with less samples and 
    more nodes

3. Check discrepency between num_samples for explicit
inference models - why is there an additional 
parameter?

4. Add better SCM distance measures?
5. Can you define perfect behavior for Entropy and 
NILL and then compute the distance between the 
observed data points and that perfect behavior?
