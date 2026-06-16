import numpy as np
from matplotlib import pyplot as plt

# q = [[0.1,0.9],[0.3,0.7],[0.4,0.6]]
q = [[0.4,0.6 ]]
q = np.array(q)
def ce(p,q):
    loss = -(np.log(p)*q).sum(1)
    return loss
def fl(p,q,gamma=1):
    loss = -((1-p)**gamma * np.log(p) * q).sum(1)
    return loss
def fl1(p,q,gamma=1):
    p = (p*q).sum(1)
    loss = -(1-p)**gamma * np.log(p)
    return loss
def fl2(p,q,gamma=1):
    p = (p*q).sum(1)
    loss = -(1-p)**gamma * np.log(p)*np.max(q,axis=1)
    return loss
def l2(p,q):
    loss = ((p-q)**2).mean(1)
    return loss
p = np.linspace(0.001,0.999,100)
q =q.repeat(100,0)
p = np.vstack([p,1-p]).transpose(1,0)
# print(ce(p,q))
# plt.plot(p[:,0],ce(p,q),label='CE=-dot(log(p),q)')
# plt.plot(p[:,0],fl(p,q),label='FL=-dot((1-p)*log(p),q)')
# plt.plot(p[:,0],fl(p,q,2),label='FL=-dot((1-p)^2*log(p),q)')
plt.plot(p[:,0],fl1(p,q),label='pt=dot(p,q), FL=-(1-pt)*log(pt)')
# plt.plot(p[:,0],fl1(p,q,2),label='pt=dot(p,q), FL=-(1-pt)^2*log(pt)')
# plt.plot(p[:,0],fl1(p,q,4),label='pt=dot(p,q), FL=-(1-pt)^4*log(pt)')
plt.plot(p[:,0],fl2(p,q),label='pt=dot(p,q), FL=-(1-pt)*log(pt)*max(q)')
plt.plot(p[:,0],l2(p,q),label='L2=mean((p-q)^2)')
plt.legend()
plt.ylabel('Loss')
plt.xlabel('p[0]')
plt.title(f'q=[{q[0,0]},{q[0,1]}]')
plt.show()