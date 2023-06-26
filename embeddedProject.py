# 참고한 코드 : https://github.com/Virtualan/musical-note-trainer/blob/bdc33a59594eab2fc02a589f81b423436fc3a156/NoteTrainer.py#L72

import RPi.GPIO as GPIO
import time
import math
import pyaudio
import pygame
from pygame.locals import *
from random import * 
import numpy as np
from scipy.signal import fftconvolve
from numpy import argmax, diff

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

do_pin = 21
doSharp_pin = 20
re_pin = 4
reSharp_pin = 12
mi_pin = 8
pa_pin = 7
paSharp_pin = 25
sol_pin = 24
solSharp_pin = 23
ra_pin = 18
raSharp_pin = 15
si_pin = 14

GPIO.setup(do_pin, GPIO.OUT) # do
GPIO.setup(doSharp_pin, GPIO.OUT) # do#
GPIO.setup(re_pin, GPIO.OUT) # re
GPIO.setup(reSharp_pin, GPIO.OUT) # re#
GPIO.setup(mi_pin, GPIO.OUT) # mi
GPIO.setup(pa_pin, GPIO.OUT) # pa
GPIO.setup(paSharp_pin, GPIO.OUT)  # pa#
GPIO.setup(sol_pin, GPIO.OUT)  # sol
GPIO.setup(solSharp_pin, GPIO.OUT) # sol#
GPIO.setup(ra_pin, GPIO.OUT) # ra
GPIO.setup(raSharp_pin, GPIO.OUT) # ra#
GPIO.setup(si_pin, GPIO.OUT) # si

#GPIO.output(do_pin, 0)
#GPIO.output(doSharp_pin, 0)
#GPIO.output(re_pin, 0)
#GPIO.output(reSharp_pin, 0)
#GPIO.output(mi_pin, 0)
#GPIO.output(pa_pin, 0)
#GPIO.output(paSharp_pin, 0)
#GPIO.output(sol_pin, 0)
#GPIO.output(solSharp_pin, 0)
#GPIO.output(ra_pin, 0)
#GPIO.output(raSharp_pin, 0)
#GPIO.output(si_pin, 0)

def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res

class SoundRecorder:
    def __init__(self):
        self.RATE=48000
        self.BUFFERSIZE=3072
        self.secToRecord=.05
        self.threadsDieNow=False
        self.newAudio=False
        
    def setup(self):
        self.buffersToRecord=int(self.RATE*self.secToRecord/self.BUFFERSIZE)
        if self.buffersToRecord==0: self.buffersToRecord=1
        self.samplesToRecord=int(self.BUFFERSIZE*self.buffersToRecord)
        self.chunksToRecord=int(self.samplesToRecord/self.BUFFERSIZE)
        self.secPerPoint=1.0/self.RATE
        self.p = pyaudio.PyAudio()
        self.inStream = self.p.open(format=pyaudio.paInt16,channels=1,rate=self.RATE,input=True,frames_per_buffer=self.BUFFERSIZE)
        self.xsBuffer=np.arange(self.BUFFERSIZE)*self.secPerPoint
        self.xs=np.arange(self.chunksToRecord*self.BUFFERSIZE)*self.secPerPoint
        self.audio=np.empty((self.chunksToRecord*self.BUFFERSIZE),dtype=np.int16)
    
    def close(self):
        self.p.close(self.inStream)
    
    def getAudio(self):
        audioString=self.inStream.read(self.BUFFERSIZE)
        self.newAudio=True
        return np.frombuffer(audioString,dtype=np.int32)
        
def parabolic(f, x): 
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)
    
def freq_from_autocorr(raw_data_signal, fs):                          
    corr = fftconvolve(raw_data_signal, raw_data_signal[::-1], mode='full')
    corr = corr[round(len(corr)/2):]
    d = diff(corr)
    start = find(d > 0)[0]
    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)
    return fs / px    

def loudness(chunk):
    data = np.array(chunk, dtype=float) / 32768.0
    ms = math.sqrt(np.sum(data ** 2.0) / len(data))
    if ms < 10e-8: ms = 10e-8
    return 10.0 * math.log(ms, 10.0)

def find_nearest(array, value):
    index = (np.abs(array - value)).argmin()
    return array[index]

def closest_value_index(array, guessValue):
    # Find closest element in the array, value wise
    closestValue = find_nearest(array, guessValue)
    # Find indices of closestValue
    indexArray = np.where(array==closestValue)
    # Numpys 'where' returns a 2D array with the element index as the value
    return indexArray[0][0]

def build_default_tuner_range():
    
    return {65.41:'C2',
            69.30:'C2#',
            73.42:'D2',
            77.78:'E2b',
            82.41:'E2',
            87.31:'F2',
            92.50:'F2#',
            98.00:'G2',
            103.80:'G2#',
            110.00:'A2',
            116.50:'A2#',
            123.50:'B2',
            130.80:'C3',
            138.60:'C3#',
            146.80:'D3',
            155.60:'E3b',
            164.80:'E3',
            174.60:'F3',
            185.00:'F3#',
            196.00:'G3',
            207.70:'G3#',
            220.00:'A3',
            233.10:'A3#',
            246.90:'B3',
            261.60:'C4',
            277.20:'C4#',
            293.70:'D4',
            311.10:'E4b',
            329.60:'E4',
            349.20:'F4',
            370.00:'F4#',
            392.00:'G4',
            415.30:'G4#',
            440.00:'A4',
            466.20:'A4#',
            493.90:'B4',
            523.30:'C5',
            554.40:'C5#',
            587.30:'D5',
            622.30:'E5b',
            659.30:'E5',
            698.50:'F5',
            740.00:'F5#',
            784.00:'G5',
            830.60:'G5#',
            880.00:'A5',
            932.30:'A5#',
            987.80:'B5',
            1047.00:'C6',
            1109.0:'C6#',
            1175.0:'D6',
            1245.0:'E6b',
            1319.0:'E6',
            1397.0:'F6',
            1480.0:'F6#',
            1568.0:'G6',
            1661.0:'G6#',
            1760.0:'A6',
            1865.0:'A6#',
            1976.0:'B6',
            2093.0:'C7'
            } 

# main func            
try:
    stepsize = 5

    # Build frequency, noteName dictionary
    tunerNotes = build_default_tuner_range()

    # Sort the keys and turn into a numpy array for logical indexing
    frequencies = np.array(sorted(tunerNotes.keys()))

    top_note = len(tunerNotes)-1
    bot_note = 0

    # Misc variables for program controls
    inputnote = 1              # the y value on the plot
    signal_level=0             # volume level
    soundgate = 19             # zero is loudest possible input level
    SR=SoundRecorder()         # recording device (usb mic)
            
    # count
    do_count = 0
    doSharp_count = 0
    re_count = 0
    reSharp_count = 0
    mi_count = 0
    pa_count = 0
    paSharp_count = 0
    sol_count = 0
    solSharp_count = 0
    ra_count = 0
    raSharp_count = 0
    si_count = 0
        
    count_to_plus = 1
    count_to_minus = 1

    while True:
        #### Main loop ####

        SR.setup()
        raw_data_signal = SR.getAudio()                                         #### raw_data_signal is the input signal data
        signal_level = round(abs(loudness(raw_data_signal)),2)                  #### find the volume from the audio sample
        # print("signal level : {}".format(signal_level))

        try:
            inputnote = round(freq_from_autocorr(raw_data_signal,SR.RATE),2)    #### find the freq from the audio sample

        except:
            inputnote = 0
        # print("inputnote : {}".format(inputnote))
        SR.close()

        if inputnote > frequencies[len(tunerNotes)-1]:                        #### not interested in notes above the notes list
            continue

        if inputnote < frequencies[0]:                                     #### not interested in notes below the notes list
            continue

        if signal_level < soundgate:                                        #### basic noise gate to stop it guessing ambient noises
            continue


        targetnote = closest_value_index(frequencies, round(inputnote, 2))      #### find the closest note in the keyed array
        # print("targetnote = {}".format(targetnote))
        note = tunerNotes[frequencies[targetnote]]
        # print("note = {}".format(note))
                    
        if note.find("1") == -1 and note.find("2") == -1: # delete noise
            if note.find("C") != -1:
                if note.find("#") != -1: # do#
                    print("found do#")
                    GPIO.output(doSharp_pin, 1)
                    doSharp_count += count_to_plus
                else: # do
                    print("found do") 
                    GPIO.output(do_pin, 1)
                    do_count += count_to_plus
            if note.find("D") != -1:
                if note.find("#") != -1: # re#
                    print("found re#")
                    GPIO.output(reSharp_pin, 1)
                    reSharp_count += count_to_plus
                else: # re
                    print("found re")
                    GPIO.output(re_pin, 1)
                    re_count += count_to_plus
            if note.find("E") != -1: # mi
                print("found mi")
                GPIO.output(mi_pin, 1)
                mi_count += count_to_plus
            if note.find("F") != -1:
                if note.find("#") != -1: # pa#
                    print("found pa#")
                    GPIO.output(paSharp_pin, 1)
                    paSharp_count += count_to_plus
                else: # pa
                    print("found pa")
                    GPIO.output(pa_pin, 1)
                    pa_count += count_to_plus
            if note.find("G") != -1:
                if note.find("#") != -1: # sol#
                    print("found sol#")
                    GPIO.output(solSharp_pin, 1)
                    solSharp_count += count_to_plus
                else: # sol
                    print("found sol")
                    GPIO.output(sol_pin, 1)
                    sol_count += count_to_plus
            if note.find("A") != -1:
                if note.find("#") != -1: # ra#
                    print("found ra#")
                    GPIO.output(raSharp_pin, 1)
                    raSharp_count += count_to_plus
                else: # ra
                    print("found ra")
                    GPIO.output(ra_pin, 1)
                    ra_count += count_to_plus
            if note.find("B") != -1: # si
                print("found si")
                GPIO.output(si_pin, 1)
                si_count += count_to_plus
                    
        if do_count <= 0:
            GPIO.output(do_pin, 0)
        else:
            GPIO.output(do_pin, 1)
            do_count -= count_to_minus
                
        if doSharp_count <= 0:
            GPIO.output(doSharp_pin, 0)
        else:
            GPIO.output(doSharp_pin, 1)
            doSharp_count -= count_to_minus
                    
        if re_count <= 0:
            GPIO.output(re_pin, 0)
        else:
            GPIO.output(re_pin, 1)
            re_count -= count_to_minus
                    
        if reSharp_count <= 0:
            GPIO.output(reSharp_pin, 0)
        else:
            GPIO.output(reSharp_pin, 1)
            reSharp_count -= count_to_minus
                        
        if mi_count <= 0:
            GPIO.output(mi_pin, 0)
        else:
            GPIO.output(mi_pin, 1)
            mi_count -= count_to_minus
                    
        if pa_count <= 0:
            GPIO.output(pa_pin, 0)
        else:
            GPIO.output(pa_pin, 1)
            pa_count -= count_to_minus
                    
        if paSharp_count <= 0:
            GPIO.output(paSharp_pin, 0)
        else:
            GPIO.output(paSharp_pin, 1)
            paSharp_count -= count_to_minus
                        
        if sol_count <= 0:
            GPIO.output(sol_pin, 0)
        else:
            GPIO.output(sol_pin, 1)
            sol_count -= count_to_minus
                        
        if solSharp_count <= 0:
            GPIO.output(solSharp_pin, 0)
        else:
            GPIO.output(solSharp_pin, 1)
            solSharp_count -= count_to_minus
                        
        if ra_count <= 0:
            GPIO.output(ra_pin, 0)
        else:
            GPIO.output(ra_pin, 1)
            ra_count -= count_to_minus
                    
        if raSharp_count <= 0:
            GPIO.output(raSharp_pin, 0)
        else:
            GPIO.output(raSharp_pin, 1)
            raSharp_count -= count_to_minus
                        
        if si_count <= 0:
            GPIO.output(si_pin, 0)
        else:
            GPIO.output(si_pin, 1)
            si_count -= count_to_minus

        time.sleep(0.08)
        
except KeyboardInterrupt:
    pass
GPIO.cleanup()
