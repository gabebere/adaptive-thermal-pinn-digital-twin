"""Crank-Nicolson solver for T_t = alpha*(T_xx + T_yy) + source."""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized

@dataclass(frozen=True)
class BoundaryConditions:
    left: callable; right: callable; bottom: callable; top: callable

def _v(f,z,t): return np.broadcast_to(np.asarray(f(z,t),float),z.shape).copy()
def _edges(T,x,y,t,bc,tol):
    l,r,b,u=_v(bc.left,y,t),_v(bc.right,y,t),_v(bc.bottom,x,t),_v(bc.top,x,t)
    for a,c,name in ((l[0],b[0],"lower-left"),(r[0],b[-1],"lower-right"),(l[-1],u[0],"upper-left"),(r[-1],u[-1],"upper-right")):
        if not np.isclose(a,c,atol=tol,rtol=tol): raise ValueError(f"Incompatible {name} corner at t={t:g}: {a} versus {c}")
    T[:,0],T[:,-1],T[0,:],T[-1,:]=l,r,b,u
    T[0,0]=(l[0]+b[0])/2; T[0,-1]=(r[0]+b[-1])/2; T[-1,0]=(l[-1]+u[0])/2; T[-1,-1]=(r[-1]+u[-1])/2

def solve_heat_2d(*,Lx,Ly,alpha,nx,ny,times,boundary,initial,source=None,corner_tolerance=1e-9):
    """Return x,y,T (shape time,y,x). Edges receive (coordinate,time)."""
    times=np.asarray(times,float)
    if nx<3 or ny<3 or min(Lx,Ly,alpha)<=0: raise ValueError("Invalid grid/physics")
    if times.ndim!=1 or len(times)<1 or times[0]!=0 or np.any(np.diff(times)<=0): raise ValueError("times must increase strictly from zero")
    x,y=np.linspace(0,Lx,nx),np.linspace(0,Ly,ny); X,Y=np.meshgrid(x,y)
    T=np.empty((len(times),ny,nx)); T[0]=np.broadcast_to(np.asarray(initial(X,Y),float),X.shape); _edges(T[0],x,y,0,boundary,corner_tolerance)
    nxi,nyi=nx-2,ny-2; dx=x[1]-x[0]; dy=y[1]-y[0]; Ix,Iy=sparse.eye(nxi),sparse.eye(nyi)
    Dx=sparse.diags((np.ones(nxi-1),-2*np.ones(nxi),np.ones(nxi-1)),(-1,0,1))/dx**2
    Dy=sparse.diags((np.ones(nyi-1),-2*np.ones(nyi),np.ones(nyi-1)),(-1,0,1))/dy**2
    L=(sparse.kron(Iy,Dx)+sparse.kron(Dy,Ix)).tocsc(); I=sparse.eye(nxi*nyi,format="csc")
    for k,(t0,t1) in enumerate(zip(times[:-1],times[1:])):
        dt=t1-t0; old,new=T[k],T[k+1]; new[:]=old; _edges(new,x,y,t1,boundary,corner_tolerance)
        rhs=(I+.5*dt*alpha*L)@old[1:-1,1:-1].ravel()
        for F in (old,new):
            q=np.zeros((nyi,nxi)); q[:,0]+=F[1:-1,0]/dx**2; q[:,-1]+=F[1:-1,-1]/dx**2; q[0,:]+=F[0,1:-1]/dy**2; q[-1,:]+=F[-1,1:-1]/dy**2
            rhs+=.5*dt*alpha*q.ravel()
        if source is not None:
            Xm,Ym=X[1:-1,1:-1],Y[1:-1,1:-1]; rhs+=.5*dt*(np.asarray(source(Xm,Ym,t0))+np.asarray(source(Xm,Ym,t1))).ravel()
        new[1:-1,1:-1]=factorized((I-.5*dt*alpha*L).tocsc())(rhs).reshape(nyi,nxi)
    return x,y,T
