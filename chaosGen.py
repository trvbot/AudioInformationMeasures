class Lorenz(PyoObject):
    """
    Chaotic attractor for the Lorenz system.

    The Lorenz attractor is a system of three non-linear ordinary differential
    equations. These differential equations define a continuous-time dynamical
    system that exhibits chaotic dynamics associated with the fractal properties
    of the attractor.

    :Parent: :py:class:`PyoObject`

    :Args:

        pitch : float or PyoObject, optional
            Controls the speed, in the range 0 -> 1, of the variations. With values
            below 0.2, this object can be used as a low frequency oscillator (LFO)
            and above 0.2, it will generate a broad spectrum noise with harmonic peaks.
            Defaults to 0.25.
        chaos : float or PyoObject, optional
            Controls the chaotic behavior, in the range 0 -> 1, of the oscillator.
            0 means nearly periodic while 1 is totally chaotic. Defaults to 0.5
        stereo, boolean, optional
            If True, 2 streams will be generated, one with the X variable signal of
            the algorithm and a second composed of the Y variable signal of the algorithm.
            These two signal are strongly related in their frequency spectrum but
            the Y signal is out-of-phase by approximatly 180 degrees. Useful to create
            alternating LFOs. Available at initialization only. Defaults to False.

    .. seealso::

        :py:class:`Rossler`, :py:class:`ChenLee`

    >>> s = Server().boot()
    >>> s.start()
    >>> a = Lorenz(pitch=.003, stereo=True, mul=.2, add=.2)
    >>> b = Lorenz(pitch=[.4,.38], mul=a).out()

    """
    def __init__(self, pitch=0.25, chaos=0.5, stereo=False, mul=1, add=0):
        pyoArgsAssert(self, "OObOO", pitch, chaos, stereo, mul, add)
        PyoObject.__init__(self, mul, add)
        self._pitch = pitch
        self._chaos = chaos
        self._stereo = stereo
        pitch, chaos, mul, add, lmax = convertArgsToLists(pitch, chaos, mul, add)
        self._base_objs = []
        self._alt_objs = []
        for i in range(lmax):
            self._base_objs.append(Lorenz_base(wrap(pitch,i), wrap(chaos,i), wrap(mul,i), wrap(add,i)))
            if self._stereo:
                self._base_objs.append(LorenzAlt_base(self._base_objs[-1], wrap(mul,i), wrap(add,i)))

    def setPitch(self, x):
        """
        Replace the `pitch` attribute.

        :Args:

            x : float or PyoObject
                new `pitch` attribute. {0. -> 1.}

        """
        pyoArgsAssert(self, "O", x)
        self._pitch = x
        x, lmax = convertArgsToLists(x)
        if self._stereo:
            [obj.setPitch(wrap(x,i)) for i, obj in enumerate(self._base_objs) if (i % 2) == 0]
        else:
            [obj.setPitch(wrap(x,i)) for i, obj in enumerate(self._base_objs)]

    def setChaos(self, x):
        """
        Replace the `chaos` attribute.

        :Args:

            x : float or PyoObject
                new `chaos` attribute. {0. -> 1.}

        """
        pyoArgsAssert(self, "O", x)
        self._chaos = x
        x, lmax = convertArgsToLists(x)
        if self._stereo:
            [obj.setChaos(wrap(x,i)) for i, obj in enumerate(self._base_objs) if (i % 2) == 0]
        else:
            [obj.setChaos(wrap(x,i)) for i, obj in enumerate(self._base_objs)]

    def ctrl(self, map_list=None, title=None, wxnoserver=False):
        self._map_list = [SLMap(0., 1., "lin", "pitch", self._pitch),
                          SLMap(0., 1., "lin", "chaos", self._chaos), SLMapMul(self._mul)]
        PyoObject.ctrl(self, map_list, title, wxnoserver)

    @property
    def pitch(self):
        """float or PyoObject. Speed of the variations."""
        return self._pitch
    @pitch.setter
    def pitch(self, x): self.setPitch(x)

    @property
    def chaos(self):
        """float or PyoObject. Chaotic behavior."""
        return self._chaos
    @chaos.setter
    def chaos(self, x): self.setChaos(x)