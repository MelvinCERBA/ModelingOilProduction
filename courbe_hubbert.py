import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# The parametrized function to be plotted
def y(t, a, b, tau):
    return a / (1+b*np.e**(-t/tau))

def yp(t, a, b, tau):
    r1 = a*b/tau
    return r1 * np.e**(-t/tau) / (1+b*np.e**(-t/tau))**2


# Define initial parameters
init_tau = 1
max_tau = 2
init_a = 3
max_a = 10
init_b = 20
max_b = 50

n = 1000
tmax = np.log(max_b*10)*max_tau
t = np.linspace(0, tmax, n)


# Create the figure and the line that we will manipulate
fig = plt.figure()
y_ax = fig.add_subplot(211)
line1, = y_ax.plot(t, y(t, init_a, init_b, init_tau), lw=2)
line1b, = y_ax.plot(t,[init_a]*n, '--r')
y_ax.set_ylim([0, max_a*1.05])
y_ax.set_xlim([0, tmax])

yp_ax = fig.add_subplot(212)
line2, = yp_ax.plot(t, yp(t, init_a, init_b, init_tau), lw=2)

yp_ax.set_ylim([0, max_a*1.05/(4*init_tau*0.8)])
yp_ax.set_xlim([0, tmax])
yp_ax.set_xlabel('Time [s]')


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
    line1.set_ydata(y(t, a_slider.val, b_slider.val, tau_slider.val))
    line1b.set_ydata([a_slider.val]*n)
    line2.set_ydata(yp(t, a_slider.val, b_slider.val, tau_slider.val))
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