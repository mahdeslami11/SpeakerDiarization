#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:42:21 2017

@author: rehan
"""
import numpy as np
import numpy
import sklearn
import matplotlib.pylab as plt
import librosa

def listOfFeatures2Matrix(features):
    '''
    listOfFeatures2Matrix(features)

    This function takes a list of feature matrices as argument and returns a single concatenated feature matrix and the respective class labels.

    ARGUMENTS:
        - features:        a list of feature matrices

    RETURNS:
        - X:            a concatenated matrix of features
        - Y:            a vector of class indeces
    '''

    X = numpy.array([])
    Y = numpy.array([])
    for i, f in enumerate(features):
        if i == 0:
            X = f
            Y = i * numpy.ones((len(f), 1))
        else:
            X = numpy.vstack((X, f))
            Y = numpy.append(Y, i * numpy.ones((len(f), 1)))
    return (X, Y)

def normalizeFeatures(features):
    '''
    This function normalizes a feature set to 0-mean and 1-std.
    Used in most classifier trainning cases.

    ARGUMENTS:
        - features:    list of feature matrices (each one of them is a numpy matrix)
    RETURNS:
        - featuresNorm:    list of NORMALIZED feature matrices
        - MEAN:        mean vector
        - STD:        std vector
    '''
    X = np.array([])

    for count, f in enumerate(features):
        if f.shape[0] > 0:
            if count == 0:
                X = f
            else:
                X = np.vstack((X, f))
            count += 1

    MEAN = np.mean(X, axis=0) + 0.00000000000001;
    STD = np.std(X, axis=0) + 0.00000000000001;

    featuresNorm = []
    for f in features:
        ft = f.copy()
        for nSamples in range(f.shape[0]):
            ft[nSamples, :] = (ft[nSamples, :] - MEAN) / STD
        featuresNorm.append(ft)
    return (featuresNorm, MEAN, STD)

def trainSVM(features, Cparam):
    '''
    Train a multi-class probabilitistic SVM classifier.
    Note:     This function is simply a wrapper to the sklearn functionality for SVM training
              See function trainSVM_feature() to use a wrapper on both the feature extraction and the SVM training (and parameter tuning) processes.
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements containt numpy matrices of features
                            each matrix features[i] of class i is [numOfSamples x numOfDimensions]
        - Cparam:           SVM parameter C (cost of constraints violation)
    RETURNS:
        - svm:              the trained SVM variable

    NOTE:
        This function trains a linear-kernel SVM for a given C value. For a different kernel, other types of parameters should be provided.
    '''

    [X, Y] = listOfFeatures2Matrix(features)
    svm = sklearn.svm.SVC(C = Cparam, kernel = 'linear',  probability = True)        
    svm.fit(X,Y)

    return svm

def smoothMovingAvg(inputSignal, windowLen=11):
    windowLen = int(windowLen)
    if inputSignal.ndim != 1:
        raise ValueError("")
    if inputSignal.size < windowLen:
        raise ValueError("Input vector needs to be bigger than window size.")
    if windowLen < 3:
        return inputSignal
    s = numpy.r_[2*inputSignal[0] - inputSignal[windowLen-1::-1], inputSignal, 2*inputSignal[-1]-inputSignal[-1:-windowLen:-1]]
    w = numpy.ones(windowLen, 'd')
    y = numpy.convolve(w/w.sum(), s, mode='same')
    return y[windowLen:-windowLen+1]

def silenceRemoval(x, Fs, stWin, stStep, smoothWindow=0.5, Weight=0.5, plot=False):
    '''
    Event Detection (silence removal)
    ARGUMENTS:
         - x:                the input audio signal
         - Fs:               sampling freq
         - stWin, stStep:    window size and step in seconds
         - smoothWindow:     (optinal) smooth window (in seconds)
         - Weight:           (optinal) weight factor (0 < Weight < 1) the higher, the more strict
         - plot:             (optinal) True if results are to be plotted
    RETURNS:
         - segmentLimits:    list of segment limits in seconds (e.g [[0.1, 0.9], [1.4, 3.0]] means that
                    the resulting segments are (0.1 - 0.9) seconds and (1.4, 3.0) seconds
    '''

    if Weight >= 1:
        Weight = 0.99
    if Weight <= 0:
        Weight = 0.01

    # Step 1: feature extraction
#    x = audioBasicIO.stereo2mono(x)                        # convert to mono
#    featTypes = [8,0,13,13]
#    ShortTermFeatures = aF.stFeatureExtraction(x, Fs, stWin * Fs, stStep * Fs,featTypes) # extract short-term features
    S = librosa.feature.melspectrogram(y=x, sr=Fs, n_fft=int(Fs*stWin), hop_length=int(Fs*stStep))
    fVects = librosa.feature.mfcc(y=x, S=librosa.power_to_db(S), sr=Fs, n_mfcc=19)
    feaEnergy = librosa.feature.rmse(y=x, frame_length=int(Fs*stWin), hop_length=int(Fs*stStep))
    ShortTermFeatures = np.append(feaEnergy,fVects,axis=0)

    # Step 2: train binary SVM classifier of low vs high energy frames
    EnergySt = ShortTermFeatures[0, :]                  # keep only the energy short-term sequence (1nd feature)
    E = numpy.sort(EnergySt)                            # sort the energy feature values:
    L1 = int(len(E) / 10)                               # number of 10% of the total short-term windows
    T1 = numpy.mean(E[0:L1]) + 0.000000000000001                 # compute "lower" 10% energy threshold
    T2 = numpy.mean(E[-L1:-1]) + 0.000000000000001                # compute "higher" 10% energy threshold
    Class1 = ShortTermFeatures[:, numpy.where(EnergySt <= T1)[0]] # get all features that correspond to low energy
    Class2 = ShortTermFeatures[:, numpy.where(EnergySt >= T2)[0]] # get all features that correspond to high energy
    featuresSS = [Class1.T, Class2.T]                             # form the binary classification task and ...

    [featuresNormSS, MEANSS, STDSS] = normalizeFeatures(featuresSS)   # normalize and ...
    SVM = trainSVM(featuresNormSS, 1.0)                  # train the respective SVM probabilistic model (ONSET vs SILENCE)

    # Step 3: compute onset probability based on the trained SVM
    ProbOnset = []
    for i in range(ShortTermFeatures.shape[1]):                    # for each frame
        curFV = (ShortTermFeatures[:, i] - MEANSS) / STDSS         # normalize feature vector
        ProbOnset.append(SVM.predict_proba(curFV.reshape(1,-1))[0][1]) # get SVM probability (that it belongs to the ONSET class)
    ProbOnset = numpy.array(ProbOnset)
    ProbOnset = smoothMovingAvg(ProbOnset, smoothWindow / stStep)  # smooth probability

    # Step 4A: detect onset frame indices:
    ProbOnsetSorted = numpy.sort(ProbOnset) # find probability Threshold as a weighted average of top 10% and lower 10% of the values
    Nt = int(ProbOnsetSorted.shape[0] / 10)
    T = (numpy.mean((1 - Weight) * ProbOnsetSorted[0:Nt]) + Weight * numpy.mean(ProbOnsetSorted[-Nt::]))
    MaxIdx = numpy.where(ProbOnset > T)[0] # get the indices of the frames that satisfy the thresholding
    i = 0
    timeClusters = []
    segmentLimits = []

    # Step 4B: group frame indices to onset segments
    while i < len(MaxIdx):                                         # for each of the detected onset indices
        curCluster = [MaxIdx[i]]
        if i == len(MaxIdx)-1:
            break
        while MaxIdx[i+1] - curCluster[-1] <= 2:
            curCluster.append(MaxIdx[i+1])
            i += 1
            if i == len(MaxIdx)-1:
                break
        i += 1
        timeClusters.append(curCluster)
        segmentLimits.append([curCluster[0] * stStep, curCluster[-1] * stStep])

    # Step 5: Post process: remove very small segments:
    minDuration = 0.2
    segmentLimits2 = []
    for s in segmentLimits:
        if s[1] - s[0] > minDuration:
            segmentLimits2.append(s)
    segmentLimits = segmentLimits2

    if plot:
        timeX = numpy.arange(0, x.shape[0] / float(Fs), 1.0 / Fs)

        plt.subplot(2, 1, 1)
        plt.plot(timeX, x)
        for s in segmentLimits:
            plt.axvline(x=s[0])
            plt.axvline(x=s[1])
        plt.subplot(2, 1, 2)
        plt.plot(numpy.arange(0, ProbOnset.shape[0] * stStep, stStep), ProbOnset)
        plt.title('Signal')
        for s in segmentLimits:
            plt.axvline(x=s[0])
            plt.axvline(x=s[1])
        plt.title('SVM Probability')
        plt.show()

    return segmentLimits, MaxIdx
