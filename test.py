from pyo import *
import deap as dp

NPARAMS = 17

def genparams():
    # used for generating randomized individuals at initialization
    params = []
    for i in range(NPARAMS):
        if i == 0:
            params.append(random.randint(1, 127))
        if i == 1:
            # appends the number of midi samples chosen
            params = params + random.sample(range(0,127), params[-1])
        if i == 2:
            params.append(random.randint(1, 4))
        if i == 3:
            for _ in range(params[-1]):
                params.append(round(random.triangular(0.033, 10, 1), 4))
        if i == 4:
            params.append(round(random.triangular(0.1, 0.9999, 0.9), 4))
        if i == 5:
            params.append(round(random.triangular(1.0001, 10, 1.1), 4))
        if i == 6:
            params.append(random.randint(1, 4))
        if i == 7:
            for _ in range(params[-1]):
                params.append(round(random.triangular(0.033, 10, 1), 4))
        if i == 8:
            # fb min
            params.append(round(random.uniform(0.0, 0.5), 4))
        if i == 9:
            # fb max
            params.append(round(random.uniform(params[-1], 1.0), 4))
        if i == 10:
            # fb update rate
            params.append(round(random.uniform(0.033, 10), 4))
        if i == 12:
            # fb update rate
            params.append(round(random.uniform(0.033, 10), 4))
        if i == 11:
            params.append(round(random.uniform(0, 10), 4))
        if i > 12:
            params.append(round(random.uniform(0, 10), 4))

    return params

def parsegenes(genes):
    ## evolvable parameters!
    # notes
    num_choices = 0     # integer between 1 and 127
    choices = []        # integers between 0 and 127
    num_rate = 0        # integer between 1 and 4
    rate = []           # floats between 1/T and 10
    # jitters
    jit_min = 0.        # float between 0.1 and 0.9999
    jit_max = 0.        # float between 1.0001 and 10
    jit_num_rate = 0    # integer between 1 and 4
    jit_rate = []       # floats between 1/T and 10
    # feedback values
    fb_min = 0.         # 0<fb_min<0.5
    fb_max = 0.         # fb_min<fb_max<1.
    fb_rate = 0.        # 1/T<fb_freq<10
    # LFO values
    sp_max = 0.         # float between 0 and 10
    sp_rate = 0.        # float between 1/T and 10
    sp_add = 0          # float between 0 and 10
    # amplitude values
    amp_mul = 0.        # float between 0 and 10
    amp_add = 0.        # float between 0 and 10
    amp_phase = 0.      # float between 0 and 10

    j = 0
    for i in range(NPARAMS):
        if i == 0:
            num_choices = genes[i]
            print('num choices = {0}, i={1}, j={2}').format(num_choices, i, j)
            j+=1
        if i == 1:
            # for as many midi samples as specified
            for _ in xrange(genes[j-1]):
                # build a list of all samples
                choices.append(genes[j])
                j += 1
            print('choices = {0}, len = {1}, i={2}, j={3}').format(choices, len(choices), i, j)
        if i == 2:
            num_rate = genes[j]
            print('num rate = {0}, i={1}, j={2}').format(num_rate, i, j)
            j += 1
        if i == 3:
            for _ in xrange(genes[j-1]):
                rate.append(genes[j])
                j += 1
            print('rate = {0}, len={1}, i={2}, j={3}').format(rate, len(rate), i, j)
        if i == 4:
            jit_min = genes[j]
            print('jitmin = {0}, i={1}, j={2}').format(jit_min, i, j)
            j += 1
        if i == 5:
            jit_max = genes[j]
            print('jitmax = {0}, i={1}, j={2}').format(jit_max, i, j)
            j += 1
        if i == 6:
            jit_num_rate = genes[j]
            print('jitnumrate = {0}, i={1}, j={2}').format(jit_num_rate, i, j)
            j += 1
        if i == 7:
            for _ in range(genes[j-1]):
                jit_rate.append(genes[j])
                j += 1
            print('jitrate = {0}, len = {1}, i={2}, j={3}').format(jit_rate, len(jit_rate), i, j)
        if i == 8:
            fb_min = genes[j]
            print('fbmin = {0}, i={1}, j={2}').format(fb_min, i, j)
            j += 1
        if i == 9:
            fb_max = genes[j]
            print('fb_max = {0}, i={1}, j={2}').format(fb_max, i, j)
            j += 1
        if i == 10:
            fb_rate = genes[j]
            print('fb_rate = {0}, i={1}, j={2}').format(fb_rate, i, j)
            j += 1
        if i == 11:
            sp_max = genes[j]
            print('sp_max = {0}, i={1}, j={2}').format(sp_max, i, j)
            j += 1
        if i == 12:
            sp_rate = genes[j]
            print('sp_rate = {0}, i={1}, j={2}').format(sp_rate, i, j)
            j += 1
        if i == 13:
            sp_add = genes[j]
            print('sp_add = {0}, i={1}, j={2}').format(sp_add, i, j)
            j += 1
        if i == 14:
            amp_mul = genes[j]
            print('amp_mul = {0}, i={1}, j={2}').format(amp_mul, i, j)
            j += 1
        if i == 15:
            amp_add = genes[j]
            print('amp_add = {0}, i={1}, j={2}').format(amp_add, i, j)
            j += 1
        if i == 16:
            amp_phase = genes[j]
            print('amp_phase = {0}, i={1}, j={2}').format(amp_phase, i, j)
            j += 1

    style = [num_choices, choices, num_rate, rate, jit_min, jit_max, jit_num_rate, jit_rate,
             fb_min, fb_max, fb_rate, sp_max, sp_rate, sp_add, amp_mul, amp_add, amp_phase]

    return style

genes = genparams()
print 'length of genes = {0}'.format(len(genes))
print genes
print '\n'
style = parsegenes(genes)
print style

