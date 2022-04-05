import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from data_processing import Data_Processing
from functions import Q, gradient_Hubbert_curve

# The parametrized function to be plotted

f = Q

# Data processing (used only to monitor the criterion function)
nameFile = "oil_france"
DP = Data_Processing(nameFile, f, gradient_Hubbert_curve) # Hubbert gradient -> temporary, not used here anyway 


X, Y = [],[]
with open("../data/{}.csv".format(nameFile),"r") as file:
    X.append(int(1974))
    Y.append(float(0))
    for line in file.readlines()[1:]:
        X.append(int(line.split(";")[5]))
        Y.append(Y[-1] +float(line.split(";")[6]))

# Define initial parameters
init_Qmax = 75000
max_Qmax = 110000

init_tmid = 17
max_tmid = 40

init_tau = 6.5
max_tau = 2*init_tau


n = 1000
tstart = X[0]
tstop = X[-1]
t = np.linspace(tstart, tstop, n)

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()

plt.scatter(X,Y,color = "red", marker = "+")

line, = plt.plot(t, f(t-tstart, (init_Qmax, init_tmid, init_tau)), lw=2)

ax.set_ylim([0, max(Y)*1.5])
ax.set_xlim([tstart, tstop])
ax.set_xlabel('Time [yr]')


# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axQmax = plt.axes([0.15, 0.25, 0.0225, 0.63])
Qmax_slider = Slider(
    ax=axQmax,
    label=r'$Qmax$',
    valmin=0.1,
    valmax=max_Qmax,
    valinit=init_Qmax,
    orientation="vertical"
)

# Make a vertically oriented slider to control the amplitude
axtau = plt.axes([0.05, 0.25, 0.0225, 0.63])
tau_slider = Slider(
    ax=axtau,
    label=r'$\tau$',
    valmin=0,
    valmax=max_tau,
    valinit=init_tau,
    orientation="vertical"
)

# Make a vertically oriented slider to control the amplitude
axtmid = plt.axes([0.25, 0.1, 0.65, 0.03])
tmid_slider = Slider(
    ax=axtmid,
    label=r'$t_{mid}$',
    valmin=0,
    valmax=max_tmid,
    valinit=init_tmid,
)

# The function to be called anytime a slider's value changes
def update(val):
    line.set_ydata(f(t-tstart, (Qmax_slider.val, tmid_slider.val, tau_slider.val)))
    print(DP.criterion((Qmax_slider.val, tmid_slider.val, tau_slider.val), 10**-6))
    fig.canvas.draw_idle()


# register the update function with each slider
tau_slider.on_changed(update)
tmid_slider.on_changed(update)
Qmax_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    tau_slider.reset()
    tmid_slider.reset()
    Qmax_slider.reset()
button.on_clicked(reset)

plt.show()
