"""Actual-cochain polynomial obstruction spaces, in exact rational arithmetic."""
from sage.all import QQ, matrix, vector
def monomials(d):
    return [(i,j,d-i-j) for i in range(d+1) for j in range(d-i+1)]
def run():
    rows=[]
    slots=[(0,0),(1,1),(0,1),(0,2),(1,2)]
    for d in range(5):
        target=monomials(d);source=monomials(d+1)
        K=matrix(QQ,5*len(target),3*len(source),sparse=True)
        for c in range(3):
            for a,alpha in enumerate(source):
                for j,power in enumerate(alpha):
                    if not power:continue
                    b=list(alpha);b[j]-=1;t=target.index(tuple(b))
                    for r,(i,k) in enumerate(slots):
                        weight=QQ(int(i==j and k==c)+int(i==c and k==j))/2
                        if i==k and j==c:weight-=QQ(1)/3
                        K[r*len(target)+t,c*len(source)+a]+=weight*power
        S=vector(QQ,K.nrows())
        # z^d diag(1,-1,0) is tracefree and divergence free in every degree.
        t=target.index((0,0,d));S[t]=1;S[len(target)+t]=-1
        rank=K.rank();aug=K.augment(matrix(QQ,list(S)).transpose()).rank()
        test=vector(QQ,[(j%5)-2 for j in range(K.ncols())])
        image=K*test
        assert K*K.solve_right(image)==image
        assert (aug==rank+1)==(d>=3)
        nonzero=sum(1 for x in S if x)
        # Compute div of the negative polynomial, term by term.
        div={}
        for i,k,coeff in [(0,0,1),(1,1,-1)]:
            powers=[0,0,d]
            if powers[i]:
                n=powers[i];powers[i]-=1
                key=(k,tuple(powers));div[key]=div.get(key,0)+coeff*n
        assert not any(div.values()) and nonzero
        rows.append(dict(degree=d,domain=int(K.ncols()),codomain=int(K.nrows()),rank=int(rank),
             augmented_rank=int(aug),image_control=True,negative_divergence_zero=True,
             negative_polynomial_nonzero=True))
    return rows
