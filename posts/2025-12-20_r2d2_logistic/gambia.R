library(geoR)

data(gambia)

Y=gambia$pos
X=scale(gambia[,c(1,2,4:8)])

## Create the random effect
s.ind <- numeric(length(Y))

for(i in 1:65){
  for(j in 1:length(Y)){
    if(X[j,2]==unique(X[,2])[i]){
      s.ind[j] = i
    }
  }
}

## Create spatial correlation matrix
X.loc <- unique(X[,1:2])

# Max distance
#r = max(C)
r = 0.986

# Drop locations
X <- X[,-(1:2)]

n=length(Y) # number of data points
p=ncol(X)# number of fixed effects
q=1 # number of random effects
L=length(unique(s.ind)) # number of levels in random effect