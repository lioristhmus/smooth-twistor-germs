"""Physical unit-sphere harmonics. Selected polar and tracefree ranges of Appendix B."""
import numpy as np
from scipy.special import expit
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu
from numpy.polynomial.legendre import legvander, legder, legval
from geometry import Grid, gram, I

def transition(x):
    """Smooth unit transition and its first two derivatives."""
    u=np.clip(x,1e-7,1-1e-7)
    v=expit(1/(1-u)-1/u)
    p=1/(1-u)**2+1/u**2
    pp=2/(1-u)**3-2/u**3
    d=v*(1-v)*p
    return v,d,v*(1-v)*((1-2*v)*p*p+pp)

class Panels(Grid):
    def __init__(self,spec,collar):
        super().__init__(spec,{'radius':1})
        self.edges=np.unique(np.r_[self.edges[self.edges<=1],np.linspace(1,2,collar+1)[1:],
                                   self.edges[self.edges>2]])
        self.half=np.diff(self.edges)/2
        self.mid=(self.edges[1:]+self.edges[:-1])/2
        self.r=(self.mid[:,None]+self.half[:,None]*self.v).ravel()
        self.dr=(self.half[:,None]*self.w).ravel()
        self.x=self.r[:,None,None]*self.n
        self.kernels=None

class System:
    """Gauss collocation with zero inner traces and an optional exterior decay condition."""
    def __init__(self,grid,A,shape,far):
        idx=grid.r>1
        self.idx=idx
        r=grid.r[idx]
        half=grid.half[grid.mid>1]
        s=len(grid.v); d=A.shape[1]; blocks=len(half)
        count=(len(r)+blocks+1)*d
        free=far is not None
        mat=lil_matrix((count+int(free),count+int(free)))
        v=np.r_[-1.,grid.v]
        basis=np.linalg.inv(legvander(v,s))
        deriv=np.column_stack([legval(grid.v,legder(c)) for c in basis.T])
        right=legvander([1.],s)[0]@basis
        self.rows=[]
        for j in range(d): mat[j,j]=1
        for n,width in enumerate(half):
            first=n*(s+1)*d
            for i in range(s):
                k=n*s+i
                row=first+(i+1)*d
                self.rows.extend(range(row,row+d))
                for j in range(d):
                    mat[row+j,first+j:first+(s+1)*d:d]=deriv[i]/width
                    for b in range(d): mat[row+j,row+b]-=A[k,j,b]/r[k]
                    if free: mat[row+j,-1]=-shape[k,j]
            boundary=first+(s+1)*d
            for j in range(d):
                mat[boundary+j,first+j:boundary+j:d]=-right
                mat[boundary+j,boundary+j]=1
        if free:mat[-1,count-d:count]=far
        self.lu=splu(mat.tocsc())
        self.dim=d;self.count=count;self.free=free
    def solve(self,F):
        n,m,d=F.shape
        rhs=np.zeros((self.count+int(self.free),m))
        rhs[self.rows]=F[self.idx].transpose(0,2,1).reshape(-1,m)
        x=self.lu.solve(rhs)
        y=np.zeros_like(F)
        y[self.idx]=x[self.rows].reshape(-1,d,m).transpose(0,2,1)
        return y,(x[-1] if self.free else np.zeros(m))

class Selected:
    def __init__(self,g,L,epsilon=.001):
        self.g=g;self.L=L;self.hs=gram(g)[:(L+1)**2]
        self.nn=g.n[...,None]*g.n[...,None,:]
        self.P=I-self.nn
        self.grad=g.dY.transpose(2,1,0)[:(L+1)**2]
        self.rot=np.cross(g.n,self.grad)
        cross=np.cross(g.n[None,:,None,:],self.hs.reshape(-1,len(g.n),3,3))
        self.hsH=(cross+cross.swapaxes(-1,-2)).reshape(len(cross),-1)/2
        r=g.r[g.r>1]; dr=g.dr[g.r>1]
        smooth,der,_=transition((r-1.5)*2)
        w=np.exp(-epsilon*smooth*np.log(r))
        eta=-epsilon*(smooth+2*r*der*np.log(r))
        self.system={};self.scalar_shift={};self.zE={};self.zH={}
        def fixed(A):return np.broadcast_to(np.array(A,dtype=float),(len(r),len(A),len(A))).copy()
        # Gauss integration from panel left to each stage, using the same polynomial nodes.
        ant=np.column_stack([legval(g.v,np.polynomial.legendre.legint(c,lbnd=-1)) for c in g.basis_inv.T])
        def primitive(v):
            v=v.reshape(-1,len(g.v));width=g.half[g.mid>1]
            total=(v@g.w)*width
            return (np.r_[0,np.cumsum(total)[:-1]][:,None]+width[:,None]*(v@ant.T)).ravel()
        for l in range(L+1):
            lam=l*(l+1);s=np.sqrt(lam);nu=s/np.sqrt(2)
            a=fixed([[-1,1],[nu*nu,-1]])
            shape=np.zeros((len(r),2));far=None
            if l:
                _,cp,cpp=transition((l*(r-1)-.1)/.9)
                cp*=l/.9;cpp*=(l/.9)**2
                weight=np.sum(dr*r**(nu+1)*cp)/(-(nu+1)*np.sum(dr*r**nu*cp))
                z=r**(nu-2)*(cp-weight*cpp)
                # Integrate once by parts for the compact shift; exactly zero outside collar.
                G=primitive((r**(nu+1)+weight*(nu+1)*r**nu)*cp)-weight*r**(nu+1)*cp
                G[(r<=1+.1/l)|(r>=1+1/l)]=0
                self.scalar_shift[l]=-w*G/(s*r*r)
                shape[:,1]=r*w*(z+eta*G/r**4)/2
                far=[nu,1]
            self.system['s',l]=System(g,a,shape,far)
            if l==0:
                self.system['e',l]=System(g,fixed([[-3]]),None,None)
            elif l==1:
                self.system['e',l]=System(g,fixed([[-3,s],[s/2,-3]]),None,None)
                self.system['h',l]=System(g,fixed([[-3]]),None,None)
            else:
                shape=np.zeros((len(r),2))
                self.zE[l]=w*r**(l-2)*cp
                shape[:,1]=self.zE[l]
                self.system['e',l]=System(g,fixed([[-3,s],[s,-2]]),shape,[l/s,1])
                nuh=np.sqrt(lam+4)
                self.zH[l]=w*r**nuh*cp
                a=fixed([[-2,-1],[-lam,2]])
                a[:,1,1]+=eta
                shape[:,0]=self.zH[l];shape[:,1]=0
                # At R the omitted source is zero; impose the decaying eigenvector.
                eigen,vec=np.linalg.eig(a[-1].T)
                self.system['h',l]=System(g,a,shape,vec[:,np.argmax(eigen)])
    def solve(self,H,M):
        g=self.g;N=(self.L+1)**2;r=g.r[:,None];idx=g.r>1
        hm=g.modes(H[...,None])[:,:N,0]
        fm=g.modes(np.sum(g.n*M,axis=-1)[...,None])[:,:N,0]
        def project(B):return (M*g.dw[None,:,None]).reshape(len(r),-1)@B.reshape(N,-1).T
        E=project(self.grad);B=project(self.rot)
        phi=np.zeros_like(hm);shift=phi.copy();a=phi.copy();e=phi.copy();b=phi.copy()
        te=phi.copy();th=phi.copy()
        for l in range(self.L+1):
            sl=slice(l*l,(l+1)**2);lam=l*(l+1);s=np.sqrt(lam)
            zero=hm[:,sl]*0
            sol,c=self.system['s',l].solve(np.stack([zero,-r*hm[:,sl]/4],axis=-1))
            phi[:,sl]=sol[...,0]
            if l:shift[idx,sl]=self.scalar_shift[l][:,None]*c/s
            if not l:
                a[:,sl]=self.system['e',l].solve(fm[:,sl,None])[0][...,0]
                continue
            sol,ce=self.system['e',l].solve(np.stack([fm[:,sl],E[:,sl]/s],axis=-1))
            a[:,sl]=sol[...,0];e[:,sl]=sol[...,1]/s
            inp=B[:,sl]/s
            if l==1:
                b[:,sl]=self.system['h',l].solve(inp[...,None])[0][...,0]/s
                continue
            solH,ch=self.system['h',l].solve(np.stack([inp,zero],axis=-1))
            b[:,sl]=solH[...,0]/s
            # d(TF Hess Y)=(2-lambda) grad Y / 2; same for its symmetric rotation.
            de=-.5*a[:,sl]-e[:,sl]
            de[idx]-=r[idx]*self.zE[l][:,None]*ce/s
            dh=(solH[...,1]-solH[...,0])/s
            dh[idx]-=r[idx]*self.zH[l][:,None]*ch/s
            te[:,sl]=de*2/(2-lam);th[:,sl]=dh*2/(2-lam)
        def vector(c,basis):return (c@basis.reshape(N,-1)).reshape(len(r),len(g.n),3)
        mixed=vector(e,self.grad)+vector(b,self.rot)
        normal=a@g.Y[:,:N].T
        correction=normal[...,None,None]*(self.nn-.5*self.P)
        correction+=mixed[...,None]*g.n[...,None,:]+g.n[...,None]*mixed[...,None,:]
        correction+=(te@self.hs+th@self.hsH).reshape(correction.shape)
        return phi@g.Y[:,:N].T,vector(shift,self.grad),correction
