import torch
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

def sinkhorn_loss_primal(x,y,epsilon,n,niter,img_dimensions) :
	

	"""
	Given two emprical measures with n points each with locations x and y 
	outputs an approximation of the OT cost with regularization parameter epsilon
	niter is the max. number of steps in sinkhorn loop
	"""
	# The Sinkhorn algorithm takes as input three variables :
	#C = _squared_distances(x, y) # Wasserstein cost function
	C = _cosine_distances(x,y) #+ _distance_cost(img_dimensions)

	mu = Variable(1./n*torch.cuda.FloatTensor(n).fill_(1),requires_grad=False) 
	nu = Variable(1./n*torch.cuda.FloatTensor(n).fill_(1),requires_grad=False)
	
	# Parameters of the Sinkhorn algorithm.
	#epsilon            = (.1)**2          # regularization parameter
	rho                = 1 #(.5) **2          # unbalanced transport (See PhD Th. of Lenaic Chizat)
	tau                = -.8               # nesterov-like acceleration
	
	lam = rho / (rho + epsilon)            # Update exponent

	# Elementary operations .....................................................................
	def ave(u,u1) : 
		"Barycenter subroutine, used by kinetic acceleration through extrapolation."
		return tau * u + (1-tau) * u1 

	def M(u,v)  : 
		"$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
		return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

	lse = lambda A    : torch.log(torch.exp(A).sum( 1, keepdim = True ) + 1e-6) # slight modif to prevent NaN
	
	# Actual Sinkhorn loop ......................................................................
	u,v,err = 0.*mu, 0.*nu, 0.
	actual_nits = 0
	
	for i in range(niter) :
		u1= u # useful to check the update
		
		u =  epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze() ) + u
		v =  epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze()) + v 
		#u = ave( u, lam * ( epsilon * ( torch.log(mu.unsqueeze(1)) - lse(M(u,v))   ) + u ) )
		#v = ave( v, lam * ( epsilon * ( torch.log(nu.unsqueeze(1)) - lse(M(u,v).t()) ) + v ) )
		err = (u - u1).abs().sum()

		actual_nits += 1
		if (err < 1e-1).data.cpu().numpy() :
			break
	U, V = u, v 
	Gamma = torch.exp( M(U,V) )            # Eventual transport plan g = diag(a)*K*diag(b)
	cost  = torch.sum( Gamma * C )         # Simplistic cost, chosen for readability in this tutorial
	
	return cost

def sinkhorn_loss_dual(x,y,epsilon,n,niter) :
	

	"""
	Given two emprical measures with n points each with locations x and y 
	outputs an approximation of the OT cost with regularization parameter epsilon
	niter is the max. number of steps in sinkhorn loop
	"""
	# The Sinkhorn algorithm takes as input three variables :
	C = _squared_distances(x, y) # Wasserstein cost function

	mu = Variable(1./n*torch.cuda.FloatTensor(n).fill_(1),requires_grad=False) 
	nu = Variable(1./n*torch.cuda.FloatTensor(n).fill_(1),requires_grad=False)
	
	# Parameters of the Sinkhorn algorithm.
	#epsilon            = (.1)**2          # regularization parameter
	rho                = 1 #(.5) **2          # unbalanced transport (See PhD Th. of Lenaic Chizat)
	tau                = -.8               # nesterov-like acceleration
	
	lam = rho / (rho + epsilon)            # Update exponent

	# Elementary operations .....................................................................
	def ave(u,u1) : 
		"Barycenter subroutine, used by kinetic acceleration through extrapolation."
		return tau * u + (1-tau) * u1 

	def M(u,v)  : 
		"$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
		return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

	lse = lambda A    : torch.log(torch.exp(A).sum( 1, keepdim = True ) + 1e-6) # slight modif to prevent NaN
	
	# Actual Sinkhorn loop ......................................................................
	u,v,err = 0.*mu, 0.*nu, 0.
	actual_nits = 0
	
	for i in range(niter) :
		u1= u # useful to check the update
		
		u =  epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze() ) + u
		v =  epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze()) + v 
		#u = ave( u, lam * ( epsilon * ( torch.log(mu.unsqueeze(1)) - lse(M(u,v))   ) + u ) )
		#v = ave( v, lam * ( epsilon * ( torch.log(nu.unsqueeze(1)) - lse(M(u,v).t()) ) + v ) )
		err = (u - u1).abs().sum()

		actual_nits += 1
		if (err < 1e-1).data.cpu().numpy() :
			break
	U, V = u, v 
	cost  = torch.sum( mu * u ) + torch.sum( nu * v )          # Simplistic cost, chosen for readability in this tutorial
	
	return cost


def _squared_distances(x, y) :
	"Returns the matrix of $\|x_i-y_j\|^2$."
	x_col = x.unsqueeze(1) #x.dimshuffle(0, 'x', 1)
	y_lin = y.unsqueeze(0) #y.dimshuffle('x', 0, 1)
	c = torch.sum( torch.abs(x_col - y_lin) , 2)
	return c 

def _cosine_distances(x,y):
    
    ### x = [batch_x,embed] , y [batch_y,embed]
    x = F.normalize(x,dim=1,p=2)
    y = F.normalize(y,dim=1,p=2)
    dot = x @ y.T ## [batch_x,batch_y]
    
    return 1-dot

def _distance_cost(img_dimensions):

  if not img_dimensions :
    print("returning zerooo")
    return 0
  x,y = img_dimensions
  line = torch.FloatTensor(range(x)).reshape(-1,1)
  column = torch.FloatTensor(range(y)).reshape(1,-1)

  a = line.repeat(1,y)
  b =column.repeat(x,1)
  cost = a+b
  print("Transport cost shape",cost.size())
  return cost.divide(x+y).reshape(-1,1)
