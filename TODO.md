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
