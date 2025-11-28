# Goals

1. each used model should be grouped into a seperate folder,
but make sure that commond model components are reused 
(e.g. if they use same decoder)
> question remains if I should just import he model code
or hardcode the architectures into the folders

    - `./bayesDag/`
    - `./avici/`
    - `./dibs/`
    - `./bcnp/`


2. Maybe introduce a wrapper with functionality for models 
to ensure same return type + trainable flag + trained flag

3. Generally, make it easy to add models to workflow and
to make other functionalities expressive, efficient and less
redundant

4. Store model artifacts (weights or posterior) in the respective
folder (or where should I best store it, together with used
data and results?)
