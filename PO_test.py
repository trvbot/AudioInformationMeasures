import POboy as p
import numpy as np
import pandas as pd
import entropy
import matplotlib.pyplot as plt


def Noise_compare():
    '''
    compares the MSE of two types of white noise
    as a validation step for the metric
    '''
    # test parameters
    T = 1
    white = p.POboy(T, manual=True)
    pink = p.POboy(T, manual=True)
    noises = [white, pink]

    # multi-scale entropy parameters
    SCALES = 100
    BINS = lambda x: 0.002*np.std(x)
    data = []

    i = 0
    sel = [0, 1]
    for P in noises:
        print('noise {0} of {1}\n').format(i+1, len(noises))
        P.translate(P.noise, [sel[i], 1, 0])
        P.flatten()
        P.condense(10)
        print('judging...\n')
        P.judge(P.memory, SCALES, BINS)
        data.append(P.MSE)
        i += 1

    return data


def Sine_compare():
    # tests effect of mul parameter
    n = 100
    min, max = 0., 10.
    value_range = np.linspace(min, max, n)
    T = 2
    sines = [p.POboy(T, manual=True) for _ in xrange(n)]

    # multi-scale entropy measures
    SCALES = 200
    BINS = lambda x: 0.002*np.std(x)
    data = []

    i = 0
    for P in sines:
        print('sine {0} of {1}\n').format(i+1, len(sines))
        P.translate(P.sine, [1000, 0, [value_range[i]], 0])
        P.flatten()
        # P.condense(5)
        print('judging...\n')
        P.judge(P.memory, SCALES, BINS)
        data.append(P.MSE)
        i += 1

    return data


def Strange_range():
    '''
    tests the output of MSE for different values of chaotic parameter
    '''

    # test parameters
    T = 2
    n = 5
    value_range = np.linspace(0., 1.0, n)
    # attractors = [p.POboy(T, manual=True) for _ in xrange(n)]

    # multi-scale entropy measures
    SCALES = 200
    BINS = lambda x: 0.002*np.std(x)
    data = []

    # parameters : pitch, chaos, mul, add

    for i in xrange(len(value_range)):
        P = p.POboy(T, manual=True)
        print('attractor {0} of {1}\n').format(i+1, len(value_range))
        P.translate(P.strange, [0.002, value_range[i], 1500, 1000])
        P.flatten()
        # P.condense(5)
        print('judging...\n')
        P.judge(P.memory, SCALES, BINS(P.memory))
        data.append(P.MSE)
        del P

    return data


def MSE_bins_test():
    '''
    tests effect of SCALES & BINS parameters

    '''

    # test parameters
    T = 1
    n = 10
    value_range = np.linspace(0.000001, 0.001, n)
    # attractors = [p.POboy(T, manual=True) for _ in xrange(n)]
    data = []
    trials = 3

    # multi-scale entropy measures
    SCALES = 100
    # BINS = lambda x, y: value_range[x]*np.std(y)

    # generate a test piece of data
    po = p.POboy(T, manual=True)
    # pink noise
    po.translate(po.noise, [1, 1, 0])
    po.flatten()
    # po.condense(10)

    print('{0} samples in sample').format(len(po.memory))
    print('of original sample, {0}\n').format(len(po.phenotype))

    for i in xrange(len(value_range)):
        print('trial {0} of {1}\n').format(i+1, len(value_range))
        print('judging...\n')
        po.judge(po.memory, SCALES, value_range[i]*np.std(po.memory))
        tmp = []
        for j in xrange(len(po.MSE)):
            tmp.append(po.MSE[j])
        # data.append([value_range[i], tmp])
        data.append(tmp)
        # trial.append(data)

    # for i in xrange(len(data[0])):
    #     for tr in xrange(trials):
    #         av = trial[tr][i]

    data = np.transpose(data)
    print value_range
    data = pd.DataFrame(data, columns=value_range).to_csv('BINS_test_data_1sec_nox_pink.csv')

    # discard class instance (and server)
    # difference between this and po.reset()??
    del po

    return data


def MSE_scales_test():
    '''
    tests effect of SCALES & BINS parameters

    '''

    # test parameters
    T = 1
    n = 1
    value_range = np.linspace(100, 1000, n, dtype=int)
    # attractors = [p.POboy(T, manual=True) for _ in xrange(n)]
    data = []

    # multi-scale entropy measures
    # SCALES = 100
    BINS = lambda x: 0.002*np.std(x)

    # generate a test piece of data
    po = p.POboy(T, manual=True)
    # pink noise
    po.translate(po.noise, [0, 1, 0])
    po.flatten()
    po.condense(10)

    for i in xrange(len(value_range)):
        print('trial {0} of {1}\n').format(i+1, len(value_range))
        print('judging...\n')
        po.judge(po.memory, value_range[i], BINS(po.memory))
        tmp = []
        for j in xrange(len(po.MSE)):
            tmp.append(po.MSE[j])
        # data.append([value_range[i], tmp])
        data.append(tmp)

    # discard class instance (and server)
    # difference between this and po.reset()??
    del po



    # # make the data nice for csv export
    # # if in dataframe already **
    # data = []
    # for i in range(len(x.iloc[:,0])):
    #     tmp = []
    #     for j in range(len(x.iloc[i,:])-1):
    #         tmp.append(x.iloc[i, j+1])
    #     data.append(tmp)

    return data


def condense_test():
    # tests effect of time-lumping data

    T = 2
    n = 5
    value_range = np.linspace(0., 10., n)
    # attractors = [p.POboy(T, manual=True) for _ in xrange(n)]

    SCALES = 100
    BINS = lambda x: 0.002*np.std(x)
    data = []

    # parameters : pitch, chaos, mul, add

    for i in xrange(len(value_range)):
        print('trial {0} of {1}\n').format(i+1, len(value_range))
        P = p.POboy(T, manual=True)
        P.translate(P.strange, [0.002, 0.5, 1500, 1000])
        P.flatten()
        P.condense(value_range[i])
        print('judging...\n')
        P.judge(P.memory, SCALES, BINS(P.memory))
        data.append(P.MSE)

    return data


def sample_duration_test():
    '''
    tests the output of MSE for different sample durations
    '''

    # test parameters
    Tmax = 5
    n = 5
    value_range = np.linspace(1., Tmax, n)
    print value_range
    # attractors = [p.POboy(T, manual=True) for _ in xrange(n)]

    # multi-scale entropy measures
    SCALES = 100
    BINS = lambda x: 0.002*np.std(x)
    data = []

    # generate test data
    for i in xrange(len(value_range)):
        P = p.POboy(value_range[i], manual=True)
        print('trial {0} of {1}\n').format(i+1, len(value_range))
        # create pink noise
        P.translate(P.noise, [1, 1, 0])
        P.flatten()
        P.condense(10)
        print('judging...\n')
        t = P.judge(P.memory, SCALES, BINS(P.memory))
        data.append([value_range[i], P.MSE, t])
        P.reset()

    return data


def fft_test():
    # measures the MSE of the fft of the phenotype

    # test parameters
    T = 5
    po = p.POboy(T, manual=True)
    data = []

    # MSE parameters
    SCALES = 100
    BINS = lambda x: 0.002*np.std(x)

    # generate test data
    po.translate(po.noise, [0, 1, 0])
    po.flatten()
    fft = np.fft.fft(po.memory)
    # po.condense(5)
    print('judging...\n')
    po.judge(fft, SCALES, BINS(fft))
    data.append(po.MSE)

    return data


def fft2_test():
    '''
    tests effect of SCALES & BINS parameters on FFT!!

    '''

    # test parameters
    T = 1
    n = 10
    value_range = np.linspace(0.0000001, 0.001, n)
    data = []
    trials = 3

    # multi-scale entropy measures
    SCALES = 100
    # BINS = lambda x, y: value_range[x]*np.std(y)

    # generate a test piece of data
    po = p.POboy(T, manual=True)
    # pink noise
    po.translate(po.noise, [1, 1, 0])
    po.flatten()
    # po.condense(10)
    x = np.fft.fft(po.memory, 100)

    print('{0} samples in sample').format(len(po.memory))
    print('of original sample, {0}\n').format(len(po.phenotype))

    for i in xrange(len(value_range)):
        print('trial {0} of {1}\n').format(i+1, len(value_range))
        print('judging...\n')
        po.judge(x, SCALES, value_range[i]*np.std(x))
        tmp = []
        for j in xrange(len(po.MSE)):
            tmp.append(po.MSE[j])
        # data.append([value_range[i], tmp])
        data.append(tmp)
        # trial.append(data)

    # for i in xrange(len(data[0])):
    #     for tr in xrange(trials):
    #         av = trial[tr][i]

    data = np.transpose(data)
    print value_range
    data = pd.DataFrame(data, columns=value_range).to_csv('fft_pink_1sec.csv')

    # discard class instance (and server)
    # difference between this and po.reset()??
    del po

    return data


def main():
    tests = [Noise_compare, Sine_compare, Strange_range, MSE_bins_test,
             MSE_scales_test, condense_test, sample_duration_test, fft_test]
    x = tests[3]()

    # plt.plot(x)

    # print x
    # pd.DataFrame(x).to_csv('data.csv')


if __name__ == '__main__':
    main()