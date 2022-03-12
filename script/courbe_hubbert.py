import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from functions import Hubbert_curve

# The parametrized function to be plotted

f = Hubbert_curve

# Data
nameFile = "oil_france"

if True:
    X, Y = [],[]
    with open("../data/{}.csv".format(nameFile),"r") as file:
        for line in file.readlines()[1:]:
            X.append(int(line.split(";")[5]))
            Y.append(float(line.split(";")[6]))

# Define initial parameters
init_tau = 7
max_tau = 2*init_tau

init_a = 4*max(Y)*init_tau
max_a = 2*init_a

init_b = np.e**((X[Y.index(max(Y))]-X[0])/init_tau)
max_b = 2*init_b


n = 1000
tmin = X[0]
tmax = X[-1]
t = np.linspace(tmin, tmax, n)

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()

plt.scatter(X,Y,color = "red", marker = "+")

line, = plt.plot(t, f(t-tmin, (init_a, init_b, init_tau)), lw=2)

ax.set_ylim([0, max(Y)*1.5])
ax.set_xlim([tmin, tmax])
ax.set_xlabel('Time [yr]')


# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axtau = plt.axes([0.25, 0.1, 0.65, 0.03])
tau_slider = Slider(
    ax=axtau,
    label=r'$\tau$',
    valmin=0.1,
    valmax=max_tau,
    valinit=init_tau,
)

# Make a vertically oriented slider to control the amplitude
axa = plt.axes([0.05, 0.25, 0.0225, 0.63])
a_slider = Slider(
    ax=axa,
    label="a",
    valmin=0,
    valmax=max_a,
    valinit=init_a,
    orientation="vertical"
)

# Make a vertically oriented slider to control the amplitude
axb = plt.axes([0.15, 0.25, 0.0225, 0.63])
b_slider = Slider(
    ax=axb,
    label="b",
    valmin=0,
    valmax=max_b,
    valinit=init_b,
    orientation="vertical"
)

# The function to be called anytime a slider's value changes
def update(val):
    line.set_ydata(f(t-tmin, (a_slider.val, b_slider.val, tau_slider.val)))
    fig.canvas.draw_idle()


# register the update function with each slider
a_slider.on_changed(update)
b_slider.on_changed(update)
tau_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    a_slider.reset()
    b_slider.reset()
    tau_slider.reset()
button.on_clicked(reset)

plt.show()