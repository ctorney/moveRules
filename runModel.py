#tester


from matplotlib import pylab as plt


import move_model
import pymc
from pymc import MCMC
from pymc.Matplot import plot as mcplot
M = MCMC(move_model)

#M.use_step_method(pymc.AdaptiveMetropolis, [M.left_angle, M.right_angle, M.lag, M.dist],  delay=1000)
M.sample(iter=20000, burn=100, thin=10,verbose=0)
#mcplot(M)
#from pylab import hist, show

plt.hist(M.trace('reprho')[:])
plt.xlim(0,1)

plt.title('repulsion strength')

plt.savefig('repulsion_strength.png')
plt.show()
plt.hist(M.trace('attrho')[:])
plt.xlim(0,1)

plt.title('attraction strength')

plt.savefig('attraction_strength.png')
plt.show()
plt.hist(M.trace('replen')[:])
plt.xlim(0,5)
plt.title('repulsion length')



plt.savefig('repulsion_length.png')
plt.show()
plt.hist(M.trace('eta')[:])
plt.xlim(0,1)

plt.title('autocorrelation')

plt.savefig('autocorrelation.png')
plt.show()

#show()





