import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.axes_grid.axislines import SubplotZero
from funcs import Hubbert_curve

# The parametrized function to be plotted

f = Hubbert_curve

# Data
nameFile = "oil_france"

hideScale=True

X, Y = [],[]
with open("../data/{}.csv".format(nameFile),"r") as file:
    for line in file.readlines()[1:]:
        X.append(int(line.split(";")[5]))
        Y.append(float(line.split(";")[6]))

# Define initial parameters
global tau, a, b, N, tmin, tmax, t, Qmax, tmid
tau = 7
a = 4*max(Y)*tau
b = 20
max_tau = 2*tau
max_a = 2*a
max_b = 2*b

Hmax = a/(4*tau)

Qmax = a
tmid = tau*np.log(b)
    
N = 1000
tmin = X[0]
tmax = X[-1]
t = np.linspace(tmin, tmax, N)



# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()

# plots the data
#plt.scatter(X,Y,color = "red", marker = "+")


plt.close()
fig = plt.figure("Courbe de Hubbert",figsize = (5,4))
ax = SubplotZero(fig, 111)
ax.set_ylim([0, max(Y)*1.5])
ax.set_xlim([tmin, tmax])
ax.set_xlabel('t')
ax.set_ylabel(r'$Hubbert_{a,b,\tau}(t)$')
fig.add_subplot(ax)
line, = plt.plot(t, f(t-tmin, (a, b, tau)), lw=2)


for direction in ["right", "top"]:
    ax.axis[direction].set_axisline_style("-|>")
    ax.axis[direction].set_visible(False)
for direction in ["left", "bottom"]:
    ax.axis[direction].set_axisline_style("-|>")
    ax.axis[direction].set_visible(True)

# turns off axis numbers if desired
if hideScale:
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    

# adjust the main plot to make room for the sliders
plt.subplots_adjust( bottom=0.30)

# Make a horizontal slider to control the frequency.
axtau = plt.axes([0.15, 0.05, 0.75, 0.03])
tau_slider = Slider(
    ax=axtau,
    label=r'$\tau$',
    valmin=0.1,
    valmax=max_tau,
    valinit=tau,
)
tau_slider.valtext.set_visible(not(hideScale))

# Make a vertically oriented slider to control the amplitude
axa = plt.axes([0.15, 0.10, 0.75, 0.03])
a_slider = Slider(
    ax=axa,
    label="a",
    valmin=0,
    valmax=max_a,
    valinit=a,
    #orientation="vertical"
)
a_slider.valtext.set_visible(not(hideScale))

# Make a vertically oriented slider to control the amplitude
axb = plt.axes([0.15, 0.15, 0.75, 0.03])
b_slider = Slider(
    ax=axb,
    label="b",
    valmin=0,
    valmax=max_b,
    valinit=b,
    #orientation="vertical"
)
b_slider.valtext.set_visible(not(hideScale))

# The function to be called anytime a slider's value changes
def update(val):
    line.set_ydata(f(t-tmin, (a_slider.val, b_slider.val, tau_slider.val)))
    fig.canvas.draw_idle()


# register the update function with each slider
a_slider.on_changed(update)
b_slider.on_changed(update)
tau_slider.on_changed(update)

# #Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
# resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
# button = Button(resetax, 'Reset', hovercolor='0.975')


# def reset(event):
#     a_slider.reset()
#     b_slider.reset()
#     tau_slider.reset()
# button.on_clicked(reset)

plt.show()