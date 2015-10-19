import numpy as np
import numpy.random as rand

def randdos(): 
	x = np.linspace(-1,1.,1000)
	pot = np.zeros([1000])
	numpeaks = 12

	for y in xrange(numpeaks):
		tpot = 0.
		wn = rand.uniform(-.7,.7)
		an = rand.exponential(1.)
		stdn = rand.uniform(0.02,0.3)
		
		#now add clustering, with random splittings 
		numsplit = 1
		an /= 2.*numsplit + 1
		stdn /= 2.*numsplit + 1
		for z in xrange(numsplit):
			sp1 = rand.exponential(stdn*an*5)
			sp2 = rand.exponential(stdn*an*5)
			an1 = an * rand.uniform(0.8,1.2)
			an2 = an * rand.uniform(0.8,1.2)
			stdn1 = stdn * rand.uniform(0.9,1.1)
			stdn2 = stdn * rand.uniform(0.9,1.1)

			tpot += an1*np.exp(-((x-wn-sp1)**2)/(stdn1**2))
			tpot += an2*np.exp(-((x-wn+sp2)**2)/(stdn2**2))

		an *= 1.2
		tpot += an*np.exp(-((x-wn)**2)/(stdn**2))
		pot += tpot
	return pot


def genparams():
	alist=[]
	wlist=[]
	stdlist=[]

	numpeaks=12
	for y in xrange(numpeaks):
		wn=rand.uniform(-.5,.5)
		an=rand.exponential(1)
		stdn=rand.uniform(0.02,0.3)

		an/=3
		stdn/=3

		sp1=rand.exponential(stdn*an*5)
		sp2=rand.exponential(stdn*an*5)
		an1=an*rand.uniform(0.8,1.2)
		an2=an*rand.uniform(0.8,1.2)
		stdn1=stdn*rand.uniform(0.9,1.1)
		stdn2=stdn*rand.uniform(0.9,1.1)

		an*=1.2

		alist.append(an)
		alist.append(an1)
		alist.append(an2)

		wlist.append(wn)
		wlist.append(wn+sp1)
		wlist.append(wn-sp2)

		stdlist.append(stdn)
		stdlist.append(stdn1)
		stdlist.append(stdn2)

	#normalization
	n=0.
	for y in xrange(len(alist)):
		n+=alist[y]*stdlist[y]

	n*=np.sqrt(np.pi)

	na=np.sqrt(n)
	nstd=na

	for y in xrange(len(alist)):
		alist[y]/=na
		stdlist[y]/=nstd

	return alist+wlist+stdlist


def normp(plist):
	q=len(plist)
	q/=3
	alist=plist[:q]
	wlist=plist[q:2*q]
	stdlist=plist[2*q:]

	n=0.
	for y in xrange(len(alist)):
		n+=alist[y]*stdlist[y]

	n*=np.sqrt(np.pi)

	na=np.sqrt(n)
	nstd=na

	for y in xrange(len(alist)):
		alist[y]/=na
		stdlist[y]/=nstd

	return alist+wlist+stdlist



def createdos(plist):
	q=len(plist)
	q/=3
	alist=plist[:q]
	wlist=plist[q:2*q]
	stdlist=plist[2*q:]

	x=np.linspace(-1,1,1000)
	pot=np.zeros([1000])

	for y in xrange(q):
		an=alist[y]
		wn=wlist[y]
		stdn=stdlist[y]
		pot += an*np.exp(-((x-wn)**2)/(stdn**2))

	# if(pot[0]>0.01 or pot[-1]>0.01):
	# 	pot = -1*np.ones(pot.shape) #in outer function, throw out neg values
	
	return x,pot


def greens(wn,plist):
	q=len(plist)
	q/=3
	alist=plist[:q]
	wlist=plist[q:2*q]
	stdlist=plist[2*q:]

	g=0.
	for x in xrange(q):
		g+=-alist[x]*np.exp(-(np.abs((1j*wn - wlist[x])/stdlist[x])**2))
	return g


from scipy.special import wofz
# blows up for iw in LHP
# put a 1j* in the argument
def Giw(z,plist):
	q=len(plist)
	q/=3
	alist=plist[:q]
	wlist=plist[q:2*q]
	stdlist=plist[2*q:]

	g=0.
	for x in xrange(q):
		g+=-1j*np.pi*alist[x]*wofz((z-wlist[x])/stdlist[x])
	return g

# can write this whole thing as a matrix vector mult.
# def numGiw(z,plist):
# 	x,pot=createdos(plist)
# 	return np.trapz(pot/(z-x),x)

# ng=zeros(w.shape,complex)
# for x in xrange(len(w)):
# 	ng[x]=numGiw(1j*w[x],plist)

def mvGiw(z,w,p):
	g=np.zeros(z.shape,complex)
	mat=np.zeros([len(z),len(p)],complex)
	for x in xrange(len(z)):
		for y in xrange(len(p)):
			mat[x,y] = 1./(z[x]-w[y])
	g=np.dot(mat,p)
	return g/500,mat/500 #need to return g/500, where coming from?



#dont put 1j* in the argument here
def Gtau(z,plist):
	gw=Giw(1j*z,plist)
	beta=2*np.pi/(z[1]-z[0])
	tau=np.linspace(0,beta,200)
	gt=0.
	for x in xrange(len(z)):
		gt+=np.real(gw[x])*np.cos(tau*z[x]) + (np.imag(gw[x])+(z[x]**-1))*np.sin(tau*z[x])
	gt*=2/beta
	# gt-=0.5 #why does this give wrong result
	return tau,gt