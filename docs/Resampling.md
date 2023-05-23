# Resampling

## Sinc interpolation

The `sample_quality` settings from 3 to 10 are interpolations based on the windowed sinc method.

<https://ccrma.stanford.edu/~jos/lumped/Windowed_Sinc_Interpolation.html>

In the RGC sfz software, we have a number of quality settings available from a drop-down menu.  
These sinc settings are labelled: 08 12 16 24 36 48 60 72  
One can easily guess that these designate a number of points. The numbers are multiples of 4, which makes them practical for SIMD processing.

For implementation purposes, we want to precompute the sinc function in a large table.
A window must be applied to the windowed sinc, in order to squash the edges.
A Kaiser window is most appropriate, which permits to keep aliasing ripples under a determined amplitude threshold. It has a parameter `Beta`, permitting to establish a compromise: the higher `Beta`, the lower is the alias magnitude, but the selectivity of the "brick wall" filter worsens near the cutoff point.

At first it seems an alright choice of `Beta` may be from about 6 to 10, as table size increases from 8 to 72.


[response.py]: tool for plotting frequency responses of sinc interpolators dynamically,
when size and `Beta` are varied, based on the `deip.pdf` paper.

<details><summary>Click here to view</summary>

```py
#!/usr/bin/env python

import numpy as np
from scipy.signal import *
from scipy.fft import *
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.figure as plfigure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import sys
#from cmd import Cmd
import tkinter as tk
from tkinter import ttk
import locale

# Set number of kernel points to compute
N = 1024

# Set type of interpolation
Interpolator = {'type': 'winsinc', 'points': 8, 'beta': 3.0}
#Interpolator = {'type': 'hermite3'}
#Interpolator = {'type': 'bspline3'}

# Nyquist frequency (only for plot, 1.0 for normalized in pi*rad/s)
NyquistF = 1.0
#NyquistF = 0.5 * 44100.0


def linear(x):
    y = 1.0-np.abs(x)
    y = np.maximum(0.0, y)
    return y

def bspline3(x):
    x = np.abs(x)
    x2 = x * x
    x3 = x2 * x
    y = 0
    p1 = (2./3.) - x2 + (1./2.) * x3
    p2 = (4./3.) - (2.) * x + x2 - (1./6.) * x3
    y = np.where(x < 2., p2, y)
    y = np.where(x < 1., p1, y)
    return y

def hermite3(x):
    x = np.abs(x)
    x2 = x * x
    x3 = x2 * x
    y = 0
    q = (5./2.) * x2
    p1 = (1.) - q + (3./2.) * x3
    p2 = (2.) - (4.) * x + q - (1./2.) * x3
    y = np.where(x < 2., p2, y)
    y = np.where(x < 1., p1, y)
    return y

def winsinc(x, beta):
    y = np.where(x == 0, 1.0, np.sin(np.pi * x) / (np.pi * x))
    y *= kaiser(len(x), beta)
    return y

###
def kernel(itp):
    type = itp['type']
    if type == 'hermite3':
        Ex = 4
        X = np.linspace(-Ex/2.0, Ex/2.0, N)
        Y = hermite3(X)
    elif type == 'bspline3':
        Ex = 4
        X = np.linspace(-Ex/2.0, Ex/2.0, N)
        Y = bspline3(X)
    elif type == 'linear':
        Ex = 2
        X = np.linspace(-Ex/2.0, Ex/2.0, N)
        Y = linear(X)
    elif type == 'winsinc':
        Ex = int(itp['points'])
        X = np.linspace(-Ex/2.0, Ex/2.0, N)
        Y = winsinc(X, float(itp['beta']))
    else:
        raise ValueError('Unknown type of interpolation')
    return Ex, X, Y

###
def plot(itp):
    print('Interpolator: %s' % (Interpolator))
    try:
        Ex, X, Y = kernel(Interpolator)
    except ValueError as e:
        print('Error:', str(e))
        return
    W, H = freqz(Y, worN=64*N, fs=1.0)

    fig = plt.gcf()
    fig.clf()
    fig.set_figwidth(15)
    ax1, ax2 = fig.subplots(2)

    ax1.grid(alpha=0.25)
    ax1.plot(X, Y)
    ax1.set_xlim(-Ex/2.0, Ex/2.0)
    ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))

    ax2.grid(alpha=0.25)
    ax2.plot(NyquistF*W*N*2.0/Ex, 20*np.log10(np.abs(H)/np.sum(Y)))
    ax2.set_xlim(0, 5*NyquistF)
    ax2.set_ylim(-120, 12)
    ax2.xaxis.set_major_locator(plticker.MultipleLocator(base=NyquistF))
    ax2.yaxis.set_major_locator(plticker.MultipleLocator(base=12.0))
    if NyquistF == 1.0:
        ax2.set_xlabel('Frequency (π rad/s)')
    else:
        ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude (dB)')

    plt.draw()

def main(args):
    plot(Interpolator)

    class Application(tk.Frame):
        def __init__(self, master=None):
            super().__init__(master)
            self.master = master
            self.pack()
            self.create_widgets()

        def create_widgets(self):
            CHOICES = ['linear', 'bspline3', 'hermite3', 'winsinc']

            self.choice = ttk.Combobox(self, values=CHOICES)
            self.choice.current(CHOICES.index(Interpolator['type']))
            self.choice.bind("<<ComboboxSelected>>", self.on_change_type)
            self.choice.pack(side="top")

            self.beta = tk.Scale(self.master, from_=1.0, to=16.0, digits=3, resolution=0.01, orient=tk.HORIZONTAL, command=self.on_change_beta)
            self.beta.set(Interpolator['beta'])
            self.beta.pack(side="top")

            self.numpoints = tk.Scale(self.master, from_=8, to=72, resolution=4, orient=tk.HORIZONTAL, command=self.on_change_numpoints)
            self.numpoints.set(Interpolator['points'])
            self.numpoints.pack(side="top")

            self.var_normalize = tk.IntVar()
            self.normalize = tk.Checkbutton(self.master, text='Normalize', command=self.on_change_normalized, variable=self.var_normalize)
            if NyquistF == 1.0:
                self.normalize.select()
            else:
                self.normalize.deselect()
            self.normalize.pack(side="top")

            # self.quit = tk.Button(self, text="QUIT", fg="red", command=self.master.destroy)
            # self.quit.pack(side="bottom")

            self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.master)
            self.canvas.get_tk_widget().pack(side="bottom")

        def on_change_type(self, event):
            Interpolator['type'] = self.choice.get()
            plot(Interpolator)

        def on_change_beta(self, value):
            Interpolator['beta'] = float(value)
            plot(Interpolator)

        def on_change_numpoints(self, value):
            Interpolator['points'] = int(value)
            plot(Interpolator)

        def on_change_normalized(self):
            value = self.var_normalize.get()
            global NyquistF
            NyquistF = (value == 0) and (0.5 * 44100) or 1.0
            plot(Interpolator)

    locale.setlocale(locale.LC_NUMERIC, 'C')
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()

###
if __name__ == '__main__':
    main(sys.argv)
```
</details>


[response.py]: https://gist.github.com/jpcima/f446ae9862965ee77b5ffa8e80882c21

