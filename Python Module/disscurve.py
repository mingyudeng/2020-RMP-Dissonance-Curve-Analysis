#!/usr/bin/env python
# coding: utf-8

#import relevant libraries
from IPython.display import Audio
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
from scipy import signal
import math 
import os.path


def plot_audio(sample_rate, impulse_response, title = ''):
    #create a new figure to display the data
    plt.figure(figsize=(20,12))
   
    #time series data
    plt.subplot(211)
    plt.title(title, fontsize = 22)
    plt.xlim([0,len(impulse_response)])
    plt.plot(impulse_response/max(impulse_response))
    plt.xticks([])
    plt.yticks(fontsize = 18)
    plt.ylabel('Amplitude', fontsize = 20)

    #spectrogram
    plt.subplot(212)
    plt.specgram(impulse_response, NFFT=2048, noverlap=1024, Fs=sample_rate)
    plt.xlabel('Seconds', fontsize = 20)
    plt.ylabel('Hertz', fontsize = 20)
    plt.ylim(ymax=20000)
    plt.yticks(fontsize = 18)

    pass



def perform_FFT(sample_rate, impulse_response, num_partials = 10, threshold = 0.05, start = 0):

    #perform FFT of audio data and extract prominent frequencies
    window_size = 32768
    fft_size = 131072

    #FFT
    fft = np.fft.rfft(impulse_response[start:start+window_size]*np.hanning(window_size), n=fft_size)
    amp = abs(fft/fft_size) #amplitudes
    amp = amp/max(amp)
    freqs = np.fft.rfftfreq(fft_size,d=1/sample_rate)

    #peak finding and sorting 
    peakIndices = signal.find_peaks_cwt(amp,np.arange(1,15))
    sortByAmplitude = np.argsort(amp[peakIndices])[::-1]
    sortedPeakIndices = [peakIndices[i] for i in sortByAmplitude]  
    sortedPeakIndices = sortedPeakIndices[0:num_partials]
    
    #pass peaks through threshold
    sortedPeakIndices = [x for x in sortedPeakIndices if amp[x] > threshold]
    
        
    peak_freqs, peak_amp = freqs[sortedPeakIndices],amp[sortedPeakIndices]
    
    return [freqs,amp,peak_freqs,peak_amp]



#this function creates a plot with prominent frequncies after performing FFT
def plot_FFT(freqs,amp,peak_freqs,peak_amp,scale = 'linear',  title = '', x_lim = 6000):
    
    #plot FFT and mark prominent frequencies
    plt.figure(figsize=(20,8))
    plt.title('Peak Detection: ' + title, fontsize = 22)
    plt.xlabel('Frequency (Hz)',fontsize = 20)
    plt.ylabel('Amplitude',fontsize = 20)
    plt.yscale(scale)
    plt.rcParams['ytick.minor.size'] = 0
    plt.rcParams['ytick.minor.width'] = 0
    plt.yticks()
    plt.xlim(0,x_lim)
    plt.yticks(fontsize = 20)
    plt.xticks(np.arange(0, x_lim+1, step=1000),fontsize = 19)
    plt.plot(freqs, amp, color = 'green')
    plt.plot(peak_freqs, peak_amp, "x", color = 'red')
    pass



#this function implements Sethares' dissonance curve algorithm
def diss_measure(peak_freqs, peak_amp, high_ratio = 4, title = '', show_ratios = True):
    
    low_ratio = 1
    inc = 0.001
    dstar = 0.24
    s1 = 0.0207
    s2 = 18.96
    c1 = 5
    c2 = -5
    a1 = -3.51
    a2 = -5.75
    n = len(peak_freqs)
    diss = []
    ratios = []
    for k in np.arange(low_ratio,high_ratio,inc):
        d = 0
        transposed = [k * i for i in peak_freqs]
        for i in range(0, n):
            for j in range(0, n):
                if transposed[j] < peak_freqs[i]:
                    fmin = transposed[j]
                else:
                    fmin = peak_freqs[i]
                s = dstar / (s1 * fmin + s2)
                fdif = abs(transposed[j] - peak_freqs[i])
                arg1 = a1 * s * fdif;
                arg2 = a2 * s * fdif;
                exp1 = math.exp(arg1)
                exp2 = math.exp(arg2)
                if peak_amp[i] < peak_amp[j]:
                    dnew = peak_amp[i] * (c1 * exp1 + c2 * exp2)
                else:
                    dnew = peak_amp[j] * (c1 * exp1 + c2 * exp2)
                d = d + dnew 
        diss.append(d)
        ratios.append(k)
        
    # determine the indices of the local minima
    minInd = []
    for i in range(0,len(diss)-1):
        if diss[i] < diss[i-1] and diss[i] < diss[i+1]:
            minInd.append(i) 
        if i == 0 and diss[i+1] > diss [i]: #for 1:1 ratio
            minInd.append(i)

    ds = np.array(diss)
    rt = np.array(ratios)
    
    fig = plt.figure(figsize=(20,8))
    dc = fig.add_subplot(111)
    dc.plot(ratios,diss, color = '#206020') 
    dc.set_xlim(xmin=1,xmax=high_ratio)
    dc.set_ylim(ymin=0)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    plt.xlabel('Frequency Ratio', fontsize=20)
    plt.ylabel('Sensory Dissonance', fontsize=20)
    
    plt.title(title, fontsize=22)
    
    dc.vlines(rt[minInd],0,ds[minInd],color = 'brown')
    dc.plot(rt[minInd],ds[minInd], "o", color = 'teal')
    
    if show_ratios == True:
        for i,j in zip(rt[minInd],ds[minInd]): #print ratios at local minima
            dc.annotate(str("{:.3f}".format(i)),xy=(i,j+0.06),rotation = 90, fontsize=12) 

    #12-tet scale axis
    tt = dc.twiny() 
    plt.xlabel('12-tet Scale Steps', fontsize=19)
    cents = [c*100 for c in range(0,int(round(12*math.log2(high_ratio)))+1)] #cent values of intervals 
    steps = [math.pow(math.pow(2,1/1200), c) for c in cents] #12-tet scale steps in frequency ratios
    step_names = ['P8','m2','M2','m3','M3','P4','d5','P5','m6','M6','m7','M7']
    
    names_used = [] #full list of step names used
    for i in range(0, int(round(12*math.log2(high_ratio)))+1):
        names_used.append(step_names[(i%12)])
    
    plt.xticks(steps, [n for n in names_used], fontsize=13)
    tt.set_xlim(1)
    dc.vlines(rt[minInd],dc.set_ylim()[1]*0.93,dc.set_ylim()[1],color = 'brown')
    
    plt.show()
    
    #outputs an numpy array with ratios and dissonance at minima
    ratios_min = rt[minInd]
    dissonances_min = ds[minInd]
    
    
    return([ratios_min, dissonances_min])    



def write_file(peak_freqs, peak_amp, ratios, dissonances, filename, savepath):
        
    #create and write in file
    complete_name = os.path.join(savepath, filename + ".txt")
    
    file1 = open(complete_name,"w")

    file1.write("frequencies, ") 
    for i in peak_freqs:
        file1.write("{:.8f}".format(i)) #write 8 digits after decimal point
        if i != peak_freqs[-1]:
            file1.write(" ")

    file1.write(';\n'+"amplitudes, ")
    for j in peak_amp:
        file1.write("{:.8f}".format(j))
        if j != peak_amp[-1]:
            file1.write(" ")

    file1.write(';\n'+"ratios, ")
    for k in ratios:
        file1.write("{:.8f}".format(k))
        if k != ratios[-1]:
            file1.write(" ")

    file1.write(';\n'+"dissonances, ")
    for w in dissonances:
        file1.write("{:.8f}".format(w))
        if w != dissonances[-1]:
            file1.write(" ")
        else:
            file1.write(";")

    file1.close()
    
    print("File saved to " + complete_name)



def write_file_direct(sample_rate, impulse_response, filename, savepath, num_partials = 10, threshold = 0.05, start = 2000, high_ratio = 4):
    
    freqs, amp, peak_freqs, peak_amp = perform_FFT(sample_rate, impulse_response, num_partials, threshold, start)
    
    #create lists of ratios and dissonances
    ratios, dissonances = diss_measure(peak_freqs, peak_amp, high_ratio)

    #create and write in file
    complete_name = os.path.join(savepath, filename + ".txt")
    
    file1 = open(complete_name,"w")

    file1.write("frequencies, ") 
    for i in peak_freqs:
        file1.write("{:.8f}".format(i)) #write 8 digits after decimal point
        if i != peak_freqs[-1]:
            file1.write(" ")

    file1.write(';\n'+"amplitudes, ")
    for j in peak_amp:
        file1.write("{:.8f}".format(j))
        if j != peak_amp[-1]:
            file1.write(" ")

    file1.write(';\n'+"ratios, ")
    for k in ratios:
        file1.write("{:.8f}".format(k))
        if k != ratios[-1]:
            file1.write(" ")

    file1.write(';\n'+"dissonances, ")
    for w in dissonances:
        file1.write("{:.8f}".format(w))
        if w != dissonances[-1]:
            file1.write(" ")
        else:
            file1.write(";")

    file1.close()
    print("File saved to " + complete_name)



def prom_freq(peak_freqs, peak_amp): 
    for i in range(peak_amp.size-1):
        if peak_amp[i] == max(peak_amp[0:]):
            return peak_freqs[i].item()

