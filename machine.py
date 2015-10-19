import numpy as np
from sklearn.kernel_ridge import KernelRidge 
import matplotlib.pyplot as plt
execfile('generatedos.py')
import scipy.spatial.distance as ssd

# square the std to get only positive variances?
# play with the kernels holding others const
# different machines for amplitudes, means, vars? does it matter?
# multistep machine--learn where peaks are, then learn amps+stds--this would just be more than one machine
# try Gtau
# tune the DoS by hand, see what changes if any happen to giw
# only output normalized DoSs?

# put in prior information? expected number of quasiparticles,

#make numpeaks variable, then learn sparse coefficients? put an L1 reg.

# think about getting it to work for simple input spaces (ie small plist), then extend
# even limit of two gaussians?
# this doesnt seem to be a good idea

#separate machines for amp, w, std?

# any other prior information?

# dos seems to work better than plist
# does not work well when have many peaks close together, but
# for isolated peaks it gets them well

# post process: normalize, low pass filter,

# is using Gtau better than Giw, since all data points are more 'equal'?

#ssd.correlation works ok
#minkowski with sig=1 and p=4 seems to work better, for amps at least
# sig smaller makes it more of a mean
# cosine is not bad either, but it does return negative values
# send the minkowski distance to LF
# make sure centered?
def kernel1(x,y):
	sig=4
	return np.exp(-ssd.minkowski(x,y,4)/sig)

# centered exp kernel?
def kernel1c(x,y):
	sig=4.
	t1=np.exp(-ssd.minkowski(x,y,4)/sig)

# this one also seems to work ok, as long as the output is normalized. why is the output so small?
def kernel2(x,y):
	sig=0.1
	return np.exp(-(ssd.minkowski(x,y,4)**2)/sig) 

#create database, try 10^4 samples, 10^2,3 first
w=np.linspace(-1,1,1000) #just take y
beta=200.

wsize=50
wn=np.array([(2*n + 1)*np.pi/beta for n in xrange(wsize)])

sampsize=(1*10)**3 
dos=np.zeros([sampsize,len(w)])
plists=np.zeros([sampsize,108])
giw=np.zeros([sampsize,2*wsize],complex)
# gtau=np.zeros([sampsize,200])

for x in xrange(sampsize):
	plist=genparams()
	y,p=createdos(plist) #if output -1 here create it again
	while(p[0]<0):
		print '-1'
		plist=genparams()
		y,p=createdos(plist)
	dos[x,:]=p
	plists[x,:]=plist
	# tau,gt=Gtau(wn,plist)
	# gtau[x,:]=gt
	g=Giw(1j*wn,plist)
	gall=np.hstack([np.real(g),np.imag(g)])
	giw[x,:]=gall
	# giw[x,:] = np.real(g)/np.imag(g)

# kr = KernelRidge(kernel='rbf',gamma=0.1)
kr = KernelRidge(kernel=kernel1)

# kr.fit(gtau,dos)
kr.fit(giw,dos) #seems to work best
# kr.fit(giw,plists)


plist=genparams()
y,p=createdos(plist)
while(p[0]<0):
	print '-1'
	plist=genparams()
	y,p=createdos(plist)
g=Giw(1j*wn,plist) #is it playing nice with the complex numbers?
gall=np.hstack([np.real(g),np.imag(g)])
# tau,gt=Gtau(wn,plist)

# predp=kr.predict(gt)[0,:]
predp=kr.predict(gall)[0,:]
# predlist=kr.predict(gall)[0,:]
# predlist=kr.predict(np.real(g)/np.imag(g))[0,:]
# y,predp=createdos(predlist)

# y,predp=createdos(predlist)
# plt.plot(w,p,w,	predp)
# plt.plot(np.arange(18),plist,'o-',np.arange(18),predlist,'o-')
# plt.figure()
plt.plot(y,p,y,predp)
plt.show()