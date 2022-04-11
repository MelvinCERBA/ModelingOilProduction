# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.axislines import SubplotZero
from matplotlib.transforms import BlendedGenericTransform

N = 1000

a = 7.5

f = lambda y: a*y**3 - 2*a*y**2 + (a+1)*y

yp = 2/3 + (1-3/a)**.5 /3
ym = 2/3 - (1-3/a)**.5 /3

pp = f(ym)
pm = f(yp)

X = np.linspace(0,yp*1.35,N)

Y = f(X)


plt.close()
fig = plt.figure("Q8_2b",figsize = (5,6))
ax = SubplotZero(fig, 111)
fig.add_subplot(ax)

plt.xticks([ym,2/3,yp],["$y_+$",r"$\frac{2}{3}$","$y_+$"])
plt.yticks([pm,pp],["$p_-$","$p_+$"])
plt.xlabel("y")
plt.ylabel("$f(y)$")

for direction in ["xzero", "yzero"]:
        ax.axis[direction].set_axisline_style("-|>")
        ax.axis[direction].set_visible(True)


for direction in ["left", "right", "bottom", "top"]:
    ax.axis[direction].set_visible(False)
    ax.axis[direction].major_ticks.set_ticksize(25)


plt.plot([ym]*N,np.linspace(0,pp*0.98,N),'--', color = "darkmagenta")
plt.plot([2/3]*N,np.linspace(0,f(2/3)*0.97,N),'--', color = "darkmagenta")
plt.plot([yp]*N,np.linspace(0,pm,N)*0.96,'--', color = "darkmagenta")

plt.plot(np.linspace(0,ym,N),[pp]*N,'--', color = "darkmagenta")
plt.plot(np.linspace(0,yp,N),[pm]*N,'--', color = "darkmagenta")

plt.plot(X,Y, color = "royalblue")
plt.scatter([yp,2/3,ym],[pm,f(2/3),pp], color = "royalblue")
plt.show()

