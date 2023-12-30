import numpy as np
from collections import deque

class LiveFilter:
    """Base class for live filters.
    """
    def process(self, x):
        # do not process NaNs
        if np.isnan(x):
            return x

        return self._process(x)

    def __call__(self, x):
        return self.process(x)

    def _process(self, x):
        raise NotImplementedError("Derived class must implement _process")


class LiveLFilter(LiveFilter):
    def __init__(self, b, a):
        """Initialize live filter based on difference equation.

        Args:
            b (array-like): numerator coefficients obtained from scipy.
            a (array-like): denominator coefficients obtained from scipy.
        """
        self.b = b
        self.a = a
        self._xs = deque([0] * len(b), maxlen=len(b))
        self._ys = deque([0] * (len(a) - 1), maxlen=len(a)-1)
    def _process(self, x):
        """Filter incoming data with standard difference equations.
        """
        self._xs.appendleft(x)
        y = np.dot(self.b, self._xs) - np.dot(self.a[1:], self._ys)
        y = y / self.a[0]
        self._ys.appendleft(y)

        return y
    

class LiveSosFilter(LiveFilter):
    """Live implementation of digital filter with second-order sections.
    """
    def __init__(self, sos):
        """Initialize live second-order sections filter.

        Args:
            sos (array-like): second-order sections obtained from scipy
                filter design (with output="sos").
        """
        self.sos = sos

        self.n_sections = sos.shape[0]
        self.state = np.zeros((self.n_sections, 2))
    def _process(self, x):
        """Filter incoming data with cascaded second-order sections.
        """
        for s in range(self.n_sections):  # apply filter sections in sequence
            b0, b1, b2, a0, a1, a2 = self.sos[s, :]

            # compute difference equations of transposed direct form II
            y = b0*x + self.state[s, 0]
            self.state[s, 0] = b1*x - a1*y + self.state[s, 1]
            self.state[s, 1] = b2*x - a2*y
            x = y  # set biquad output as input of next filter section.

        return y

if __name__ == '__main__':

    import scipy.signal
    from sklearn.metrics import mean_absolute_error as mae
    import matplotlib.pyplot as plt

    np.random.seed(42)  # for reproducibility
    # create time steps and corresponding sine wave with Gaussian noise
    fs = 30  # sampling rate, Hz
    ts = np.arange(0, 5, 1.0 / fs)  # time vector - 5 seconds

    ys = np.sin(2*np.pi * 1.0 * ts)  # signal @ 1.0 Hz, without noise
    yerr = 0.5 * np.random.normal(size=len(ts))  # Gaussian noise
    yraw = ys + yerr


    # define lowpass filter with 2.5 Hz cutoff frequency
    b, a = scipy.signal.iirfilter(4, Wn=2.5, fs=fs, btype="low", ftype="butter")
    y_scipy_lfilter = scipy.signal.lfilter(b, a, yraw)

    live_lfilter = LiveLFilter(b, a)
    # simulate live filter - passing values one by one
    y_live_lfilter = [live_lfilter(y) for y in yraw]

    print(f"lfilter error: {mae(y_scipy_lfilter, y_live_lfilter):.5g}")


    plt.figure(figsize=[6.4, 2.4])
    plt.plot(ts, yraw, label="Noisy signal")
    plt.plot(ts, y_scipy_lfilter, lw=2, label="SciPy lfilter")
    plt.plot(ts, y_live_lfilter, lw=4, ls="dashed", label="LiveLFilter")

    plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1], ncol=3,
            fontsize="smaller")
    plt.xlabel("Time / s")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
