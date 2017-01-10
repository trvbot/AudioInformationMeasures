from pyo import *
import time
import soundfile as sf
import numpy as np
import entropy
import matplotlib.pyplot as plt

"""
 :parameters:
    >how many midi samples (cap at 1 initially)
    >which midi samples (generate a list of samples (integers)
    >how many jitters (generate an integer)
    >what the jitter freq is
    >what min and max are (centered on 1, m


"""

NPARAMS = 17

# max time for phrases (in sec)
T = 10

class POboy(object):
    def __init__(self, T, manual):
        self.T = T

        ## evolvable parameters!
        # notes
        self.num_choices = 0     # integer between 1 and 127
        self.choices = []        # integers between 0 and 127
        self.num_rate = 0        # integer between 1 and 4
        self.rate = []           # floats between 1/T and 10
        # jitters
        self.jit_min = 0.        # float between 0.1 and 0.9999
        self.jit_max = 0.        # float between 1.0001 and 10
        self.jit_num_rate = 0    # integer between 1 and 4
        self.jit_rate = []       # floats between 1/T and 10
        # feedback values
        self.fb_min = 0.         # 0<fb_min<0.5
        self.fb_max = 0.         # fb_min<fb_max<1.
        self.fb_rate = 0.        # 1/T<fb_freq<10
        # LFO values
        self.sp_max = 0.         # float between 0 and 10
        self.sp_rate = 0.        # float between 1/T and 10
        self.sp_add = 0          # float between 0 and 10
        # amplitude values
        self.amp_mul = 0.        # float between 0 and 10
        self.amp_add = 0.        # float between 0 and 10
        self.amp_phase = 0.      # float between 0 and 10

        self.genes = []
        self.style = [self.num_choices, self.choices, self.num_rate, self.rate, self.jit_min, self.jit_max,
                     self.jit_num_rate, self.jit_rate, self.fb_min, self.fb_max, self.fb_rate,
                     self.sp_max, self.sp_rate, self.sp_add, self.amp_mul, self.amp_add, self.amp_phase]

        self.RNA = []
        self.phenotype = []
        self.memory = []

        self.archive = []

        self.s = Server().boot()
        self.voice = []

        self.sine_DNA = self.initialize_gene(self.sine_gene())
        self.strange_DNA = self.initialize_gene(self.strange_gene())

        if manual:
            self.manual = True
        else:
            self.manual = False

    def genparams(self):
        # used for generating randomized individuals at initialization
        for i in range(NPARAMS):
            if i == 0:
                self.genes.append(random.randint(1, 35))
            if i == 1:
                # appends the number of midi samples chosen
                self.genes = self.genes + random.sample(range(0,127), self.genes[-1])
            if i == 2:
                self.genes.append(random.randint(1, 4))
            if i == 3:
                for _ in range(self.genes[-1]):
                    self.genes.append(round(random.triangular(0.033, 10, 1), 4))
            if i == 4:
                self.genes.append(round(random.triangular(0.1, 0.9999, 0.9), 4))
            if i == 5:
                self.genes.append(round(random.triangular(1.0001, 10, 1.1), 4))
            if i == 6:
                self.genes.append(random.randint(1, 4))
            if i == 7:
                for _ in range(self.genes[-1]):
                    self.genes.append(round(random.triangular(0.033, 10, 1), 4))
            if i == 8:
                # fb min
                self.genes.append(round(random.uniform(0.0, 0.5), 4))
            if i == 9:
                # fb max
                self.genes.append(round(random.uniform(self.genes[-1], 1.0), 4))
            if i == 10:
                # fb update rate
                self.genes.append(round(random.uniform(0.033, 10), 4))
            if i == 12:
                # fb update rate
                self.genes.append(round(random.uniform(0.033, 10), 4))
            if i == 11:
                self.genes.append(round(random.uniform(0, 10), 4))
            if i > 12:
                self.genes.append(round(random.uniform(0, 10), 4))

    def parsegenes(self, *kwgs):
        if 'RNA' in kwgs: self.genes = RNA

        j = 0
        for i in range(NPARAMS):
            if i == 0:
                self.num_choices = self.genes[i]
                print('num choices = {0}, i={1}, j={2}').format(self.num_choices, i, j)
                j+=1
            if i == 1:
                # for as many midi samples as specified
                for _ in xrange(self.genes[j-1]):
                    # build a list of all samples
                    self.choices.append(self.genes[j])
                    j += 1
                print('choices = {0}, len = {1}, i={2}, j={3}').format(self.choices, len(self.choices), i, j)
            if i == 2:
                self.num_rate = self.genes[j]
                print('num rate = {0}, i={1}, j={2}').format(self.num_rate, i, j)
                j += 1
            if i == 3:
                for _ in xrange(self.genes[j-1]):
                    self.rate.append(self.genes[j])
                    j += 1
                print('rate = {0}, len={1}, i={2}, j={3}').format(self.rate, len(self.rate), i, j)
            if i == 4:
                self.jit_min = self.genes[j]
                print('jitmin = {0}, i={1}, j={2}').format(self.jit_min, i, j)
                j += 1
            if i == 5:
                self.jit_max = self.genes[j]
                print('jitmax = {0}, i={1}, j={2}').format(self.jit_max, i, j)
                j += 1
            if i == 6:
                self.jit_num_rate = self.genes[j]
                print('jitnumrate = {0}, i={1}, j={2}').format(self.jit_num_rate, i, j)
                j += 1
            if i == 7:
                for _ in range(self.genes[j-1]):
                    self.jit_rate.append(self.genes[j])
                    j += 1
                print('jitrate = {0}, len = {1}, i={2}, j={3}').format(self.jit_rate, len(self.jit_rate), i, j)
            if i == 8:
                self.fb_min = self.genes[j]
                print('fbmin = {0}, i={1}, j={2}').format(self.fb_min, i, j)
                j += 1
            if i == 9:
                self.fb_max = self.genes[j]
                print('fb_max = {0}, i={1}, j={2}').format(self.fb_max, i, j)
                j += 1
            if i == 10:
                self.fb_rate = self.genes[j]
                print('fb_rate = {0}, i={1}, j={2}').format(self.fb_rate, i, j)
                j += 1
            if i == 11:
                self.sp_max = self.genes[j]
                print('sp_max = {0}, i={1}, j={2}').format(self.sp_max, i, j)
                j += 1
            if i == 12:
                self.sp_rate = self.genes[j]
                print('sp_rate = {0}, i={1}, j={2}').format(self.sp_rate, i, j)
                j += 1
            if i == 13:
                self.sp_add = self.genes[j]
                print('sp_add = {0}, i={1}, j={2}').format(self.sp_add, i, j)
                j += 1
            if i == 14:
                self.amp_mul = self.genes[j]
                print('amp_mul = {0}, i={1}, j={2}').format(self.amp_mul, i, j)
                j += 1
            if i == 15:
                self.amp_add = self.genes[j]
                print('amp_add = {0}, i={1}, j={2}').format(self.amp_add, i, j)
                j += 1
            if i == 16:
                self.amp_phase = self.genes[j]
                print('amp_phase = {0}, i={1}, j={2}').format(self.amp_phase, i, j)
                j += 1

    def play(self):

        mid = Choice(choice=self.choices, freq=self.rate)
        jit = Randi(min=self.jit_min, max=self.jit_max, freq=self.jit_rate)
        fr = MToF(mid, mul=jit)
        fb = Randi(min=self.fb_min, max=self.fb_max, freq=self.fb_rate)
        sp = RandInt(max=self.sp_max, freq=self.sp_rate, add=self.sp_add)
        amp = Sine(sp, mul=self.amp_mul, add=self.amp_add, phase=self.amp_phase)
        a = SineLoop(freq=fr, feedback=fb, mul=amp).out()
        self.s.gui(locals())

    def record(self):
        self.s.recordOptions(dur=T, filename='PO.wav', fileformat=0, sampletype=1)

        self.s.recstart()
        self.s.start()
        start = time.time()

        while time.time()-start < self.T+0.1:
            mid = Choice(choice=self.choices, freq=self.rate)
            jit = Randi(min=self.jit_min, max=self.jit_max, freq=self.jit_rate)
            fr = MToF(mid, mul=jit)
            fb = Randi(min=self.fb_min, max=self.fb_max, freq=self.fb_rate)
            sp = RandInt(max=self.sp_max, freq=self.sp_rate, add=self.sp_add)
            amp = Sine(sp, mul=self.amp_mul, add=self.amp_add, phase=self.amp_phase)
            a = SineLoop(freq=fr, feedback=fb, mul=amp).out()

        self.s.stop()

        self.phenotype, sr = sf.read('PO.wav')

    def condense(self, n):
        # to compress huge time series data
        # places cutoff at ~4kHz if /5

        # made so that you may continue to produce differently condensed
        # recordings from the same phenotype

        if n == 0: return

        # n is the reduction factor
        num_bins = len(self.phenotype)/n
        y = np.zeros(num_bins)

        for x in range(num_bins):
            y[x] = np.mean(self.phenotype[x:x+n-1])

        self.memory = y

    def judge(self, data, scales, bins):
        start = time.time()
        self.MSE = entropy.multiscale_entropy(data, scales, bins)

        # filter MSE so inf appears as 10
        # for i in range(len(self.MSE)):
        #     if self.MSE[i] > 10:
        #         self.MSE[i] = 10.0

        t = time.time()-start
        print('MSE is: {0}\n').format(self.MSE)
        print('took {0} sec\n').format(t)

        return t

    def demo(self):
        # Two streams of midi pitches chosen randomly in a predefined list.
        # The argument `choice` of Choice object can be a list of lists to
        # list-expansion.
        # mid = Choice(choice=[60,62,63,65,67,69,71,72], freq=[2,3])
        # list of choices, frequency of draw, mul, and add
        mid = Choice(choice=[50, 53, 55, 58], freq=[1])

        # x[n] = (r + 3) * x[n-1] * (1.0 - x[n-1])
        # LogiMap(chaos=0.6, freq=1.0, init=0.5, mul=1, add=0)

        # Two small jitters applied on frequency streams.
        # Randi interpolates between old and new values.
        jit = Randi(min=0.99, max=2.5, freq=[6, 1, 0.04])

        # Converts midi pitches to frequencies and applies the jitters.
        fr = MToF(mid, mul=jit)

        # Chooses a new feedback value, between 0 and 0.15, every 4 seconds.
        fd = Randi(min=0, max=0.15, freq=0.25)

        # RandInt generates a pseudo-random integer number between 0 and `max`
        # values at a frequency specified by `freq` parameter. It holds the
        # value until the next generation.
        # Generates an new LFO frequency once per second.
        sp = RandInt(max=6, freq=1, add=8)

        # Creates an LFO oscillating between 0 and 0.4.
        #   mul is the amplitude
        #   add is the offset
        amp = Sine(sp, mul=0.2, add=0.2)

        a = SineLoop(freq=fr, feedback=fd, mul=amp).out()

        self.s.gui(locals())

    def load(self):
        # notes
        self.num_choices = 6                    # integer between 1 and 127
        self.choices = [50, 52, 54, 55, 57, 58] # integers between 0 and 127
        self.num_rate = 3                       # integer between 1 and 4
        self.rate = [2, 3, 4]                   # floats between 1/T and 10
        # jitters
        self.jit_min = 0.95                 # float between 0.1 and 0.9999
        self.jit_max = 1.0                  # float between 1.0001 and 10
        self.jit_num_rate = 3               # integer between 1 and 4
        self.jit_rate = [1.5, 0.3, 0.04]    # floats between 1/T and 10
        # feedback values
        self.fb_min = 0.0               # 0<fb_min<0.5
        self.fb_max = 0.2               # fb_min<fb_max<1.
        self.fb_rate = 0.25             # 1/T<fb_freq<10
        # LFO values
        self.sp_max = 6                 # float between 0 and 10
        self.sp_rate = 1                # float between 1/T and 10
        self.sp_add = 8                 # float between 0 and 10
        # amplitude values
        self.amp_mul = 0.2              # float between 0 and 10
        self.amp_add = 0.2              # float between 0 and 10
        self.amp_phase = 0.0            # float between 0 and 10

    def plotMSE(self):
        axes = plt.gca()
        axes.set_ylim(0, 10)
        plt.plot(self.MSE)

    def plotPheno(self):
        axes = plt.gca()
        axes.set_ylim(-1, 1)
        plt.plot(self.phenotype)

    def plotMem(self):
        plt.plot(self.memory)

    def reset(self):
        # notes
        self.num_choices = 0     # integer between 1 and 127
        self.choices = []        # integers between 0 and 127
        self.num_rate = 0        # integer between 1 and 4
        self.rate = []           # floats between 1/T and 10
        # jitters
        self.jit_min = 0.        # float between 0.1 and 0.9999
        self.jit_max = 0.        # float between 1.0001 and 10
        self.jit_num_rate = 0    # integer between 1 and 4
        self.jit_rate = []       # floats between 1/T and 10
        # feedback values
        self.fb_min = 0.         # 0<fb_min<0.5
        self.fb_max = 0.         # fb_min<fb_max<1.
        self.fb_rate = 0.        # 1/T<fb_freq<10
        # LFO values
        self.sp_max = 0.         # float between 0 and 10
        self.sp_rate = 0.        # float between 1/T and 10
        self.sp_add = 0          # float between 0 and 10
        # amplitude values
        self.amp_mul = 0.        # float between 0 and 10
        self.amp_add = 0.        # float between 0 and 10
        self.amp_phase = 0.      # float between 0 and 10

        self.genes = []
        self.style = [self.num_choices, self.choices, self.num_rate, self.rate, self.jit_min, self.jit_max,
                     self.jit_num_rate, self.jit_rate, self.fb_min, self.fb_max, self.fb_rate,
                     self.sp_max, self.sp_rate, self.sp_add, self.amp_mul, self.amp_add, self.amp_phase]

        self.phenotype = []
        self.memory = []

        self.voice = []

        self.s.reinit()

    def flatten(self):
        # erases stereo, (just keeps one channel)
        self.memory = self.phenotype[:,0]

    def test(self):
        # generates visualization of initialized params
        self.archive = []

        for i in range(10):
            print('>>{0}').format(i+1)
            self.reset()
            self.genparams()
            self.parsegenes()
            print 'recording'
            self.record(2)
            print 'compressing'
            self.flatten()
            self.condense(10)
            print 'judging'
            self.judge(self.memory, 100, 0.002*np.std(self.memory))
            # print 'plotting'
            # self.plotMSE()
            print 'saving'
            self.archive.append(self.MSE)
            print '\n'

        for i in range(10):
            plt.plot(self.archive[i])

    def GOrandom(self):
        self.reset()
        self.genparams()
        self.parsegenes()
        self.play()

    def translate(self, TF, RNA):
        self.s.recordOptions(dur=T, filename='PO.wav', fileformat=0, sampletype=1)

        self.s.recstart()
        self.s.start()
        start = time.time()

        while time.time()-start < self.T+0.1:
            self.voice = TF(RNA).out()

        self.s.stop()

        self.phenotype, sr = sf.read('PO.wav')

    def initialize_gene(self, DNA):
        RNA = []

        for i in range(len(DNA)):
            if type(DNA[i]['max']) == int:
                RNA.append([str(DNA[i]['name']), int(random.uniform(DNA[i]['min'], DNA[i]['max']))])
            if type(DNA[i]['max']) == float:
                RNA.append([str(DNA[i]['name']), round(random.uniform(DNA[i]['min'], DNA[i]['max']), 4)])

        return dict(RNA)

    def noise_gene(self):
        DNA = []

        name_list = ['sel', 'mul', 'add']
        min_list = [0, 0., 0.]
        max_list = [2, 10., 10.]

        for i in range(3):
            DNA.append({'name':name_list[i], 'min':min_list[i], 'max':max_list[i]})

        return DNA

    def noise(self, RNA):
        if self.manual:
            s = RNA[0]
            m = RNA[1]
            a = RNA[2]

        else:
            s = RNA['sel']
            m = RNA['mul']
            a = RNA['add']

        # White noise, pink noise, brown noise
        n = [Noise(m, a), PinkNoise(m, a), BrownNoise(m, a)]

        return n[s]

    def sine_gene(self):
        DNA = []

        name_list = ['freq', 'phase', 'mul', 'add']
        min_list = [1./T, -180., 0., 0.]
        max_list = [22000., 180., 10., 10.]

        for i in range(4):
            DNA.append({'name':name_list[i],'min':min_list[i], 'max':max_list[i]})

        return  DNA

    def sine(self, RNA):
        if self.manual:
            f = RNA[0]
            p = RNA[1]
            m = RNA[2]
            a = RNA[3]

        else:
            f = RNA['freq']
            p = RNA['phase']
            m = RNA['mul']
            a = RNA['add']

        return Sine(freq=f, phase=p, mul=m, add=a)

    def strange_gene(self):
        DNA = []

        name_list = ['pitch', 'chaos', 'mul', 'add']
        min_list = [0., 0., 250., 500.]
        max_list = [1., 1., 2000., 2000.]

        for i in range(4):
            DNA.append({'name':name_list[i],'min':min_list[i], 'max':max_list[i]})

        return  DNA

    def strange(self, RNA):
        if self.manual:
            p = RNA[0]
            c = RNA[1]
            m = RNA[2]
            a = RNA[3]
        else:
            p = RNA['pitch']
            c = RNA['chaos']
            m = RNA['mul']
            a = RNA['add']

        # choose type of chaotic attractor
        voice = [Rossler, Lorenz, ChenLee]
        n = 2

        voice = voice[n](pitch=p, chaos=c, mul=m, add=a, stereo=True)

        # or Sine(voice, mul=1).out()
        return Sine(voice, mul=0.3).out()


def fourier(data):
    np.fft.fft(data, 44100)

    return

def main():
    po = POboy(T)
    # po.genparams()
    # po.parsegenes()
    # po.play()
    po.load()

    po.record()

    po.flatten()
    # n is reduction factor
    po.condense(10)


    po.plotMem()

    scales = 200
    bins = 0.002*np.std(po.memory)
    po.judge(po.memory, scales, bins)

    po.plotMSE()

def main2():
    po = POboy(T)
    po.translate(po.sine, po.initialize_gene(po.sine_gene()))
    # po.plotPheno()

    F = np.fft.fft(po.phenotype)

    po.judge(po.phenotype, 100, 0.002*np.std(po.phenotype))

if __name__ == '__main__':
        main2()