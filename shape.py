import numpy as np
import matplotlib.pyplot as plt

L = 2
D = 1

Nm = 4 # coils per mirror part
Ns = 8 # coils per stellarator half

temp = np.linspace(-L/2,L/2,Nm+1)
Mx = (temp[1:] + temp[:-1])/2

temp = np.linspace(-np.pi/2, np.pi/2, Ns+1)
St = (temp[1:] + temp[:-1])/2

points = []
for x in Mx:
    points.append([x,D/2])

for t in St:
    points.append([L/2 + D/2 * np.cos(t), D/2 * np.sin(t)])

for x in Mx:
    points.append([x,-D/2])

for t in St:
    points.append([-L/2 - D/2 * np.cos(t), D/2 * np.sin(t)])


print(points)
x,y = np.transpose(points)
plt.plot(x,y,'.')
plt.axis('equal')
import pdb
pdb.set_trace()
plt.show()
