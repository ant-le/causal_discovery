# Goals

1. Establish Learning (Train, Optimization, MCMC) methods for
all models based on trainable flag

2. Establish (same) Inference methods for evaluating all models

3. Ensure good parallelization strategies for handling 
data/models in parallel (GPU)
    - each model sits on different GPU processing same data 
    generated on the fly (probably more efficient and easier
    than parallel data)
