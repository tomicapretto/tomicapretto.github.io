The "first" R2D2 prior

- Where it appeared
- In the context of linear regression models, it allows to put a prior on an interpretable model quantity (coefficient of determination)
    - Formula for the coefficient of determination
- Which implicitly puts a prior on the total variance of the regression parameters (variability of the linear predictor)
- That total variability is distributed to regression coefficients via a Dirichlet Decomposition
- It fits within the "global-local shrinkage prior framework".

R2D2 for GLMMs

- What GLMMs look like
- The global variance parameter
- What R^2 looks like in this case
    - General definition, variance decomposition
    - What the authors use
- They propose a prior on the global variance parameter that implies a Beta prior on R^2
    - There are closed form expressions for certain GLMMs
    - There's a general approximation: put a Generalized Beta Prime distribution on W
    - Which GBP on W for R^2 ~ Beta(a, b)? W ~ GBP(a*, b*, c*, d*)
    - The paramters a*, b*, c*, d* are found with an optimization process.
        - a* and b* are not necessarily similar to 'a' and 'b'
- R2D2 for Logistic regression
    - Application 1: simulated data
    - Application 2: real world case
