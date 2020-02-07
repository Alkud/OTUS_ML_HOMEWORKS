import numpy as np
import scipy.signal as ssgn

import essentia.standard as es

import pywt

##########################################
#           Utility functions            #
##########################################

def get_mfcc(frames, sample_rate=16000, num_bands=64, num_coeffs=32, window_type='hann'):
    '''
    Calculates amplitude spectrum, mel-frequency spectrum and mel-frequency cepstral coefficients.

    Parameters:
    frames          : overlapping signal frames for short-time analysis
    sample_rate     : audio sampling rate,
    num_bands       : number of mel-frequency bands
    num_coeffs      : number of mel-freq cepstrum coefficients
    window_type     : type of windowing function to apply

    Returns three 2D numpy arrays: amplitude spectra, mel-freq spectra and MFCCs
    '''
    
    frame_size = len(frames[0])
    spectra = []
    melbands = []
    mfccs = []

    spectrum_estimator = es.Spectrum(size=frame_size)
    windowing = es.Windowing(type='hann', size=frame_size)
    mfcc_estimator = es.MFCC(
        numberBands=num_bands, numberCoefficients=num_coeffs+1,
        inputSize=frame_size,
        sampleRate=sample_rate, highFrequencyBound=8000
    )

    for frame in frames:    
        spectrum = spectrum_estimator(windowing(frame))
        mfcc_bands, mfcc_coeffs = mfcc_estimator(spectrum)
        spectra.append(spectrum)
        mfccs.append(mfcc_coeffs[1:])
        melbands.append(mfcc_bands)
    
    return np.array(spectra).T, np.array(melbands).T, np.array(mfccs).T

def get_lpc(frames, sample_rate=16000, num_coeffs=32, window_type='hann'):
    '''
    Calculates linear prediction coefficients

    Parameters:
    frames          : overlapping signal frames for short-time analysis
    sample_rate     : audio sampling rate,    
    num_coeffs      : number of linear prediction coefficients
    window_type     : type of windowing function to apply

    Returns two numpy 2D arrays: LPCs and reflection coefficients
    '''    
    frame_size = len(frames[0])
    lpc_coeffs = []
    reflection_coeffs = []
    
    lpc_estimator = es.LPC(sampleRate=sample_rate, order=num_coeffs-1)
    windowing = es.Windowing(type='hann', size=frame_size)
    
    for frame in frames:    
        lpc, reflection = lpc_estimator(windowing(frame) * 1000)
        lpc_coeffs.append(lpc)
        reflection_coeffs.append(reflection)
    
    return np.array(lpc_coeffs).T, np.array(reflection_coeffs).T

def get_constantq(frames, sample_rate=16000, num_bands=64):
    max_freq = 8000
    min_freq = 125
    num_octaves = np.log2(max_freq / min_freq)
    bins_per_octave = int(np.ceil(num_bands / num_octaves))
    
    frame_size = len(frames[0])
    const_q_spectra = []

    spectrum_estimator = es.Spectrum(size=frame_size)
    if num_bands==16:
        padding_size = max([0, 512 - frame_size])
    elif num_bands==32:
        padding_size = max([0, 2048 - frame_size])
    else:
        padding_size = max([0, 1024 - frame_size])
        
    windowing = es.Windowing(type='hann', size=frame_size, zeroPadding=padding_size)
    
    constantq_estimator = es.ConstantQ(
        binsPerOctave=bins_per_octave,
        minFrequency=min_freq,
        numberBins=num_bands,
        sampleRate=sample_rate
    )
    for frame in frames:    
        const_q_spectrum = constantq_estimator(windowing(frame))  
        const_q_spectra.append(np.abs(const_q_spectrum))
    
    return np.array(const_q_spectra).T

def compress_range(x, factor=16):
    N = len(x)
    indices = []
    for i in range(int(np.log2(N)*factor)):
        indices.append(int(np.power(2, i/factor)))
    indices.append(N-1)
    output = []
    for i in range(1,len(indices)):
        output.append(np.max(x[indices[i] : indices[i] + 1]))
    return output

def get_spectrum_envelopes(lpc, sample_rate=16000, num_bands=64):
    '''
    Takes an array of LPC and calculates corresponding frequency responses equal to spectrum envelopes.
    Returns spectrum envelopes in log scale for both magnitude and frequency ranges.
    '''
    spectrum_envelopes = []
    num_lin_freqs = int(np.ceil(np.power(2, num_bands/5)))    
    for lp_coeffs in lpc.T:
        freqs, complex_transfer = ssgn.freqz(1, lp_coeffs, worN=num_lin_freqs, fs=sample_rate)
        log_power_spectrum = np.abs(complex_transfer)  ## np.log10(np.abs(complex_transfer)+1)**2
        log_freq = compress_range(log_power_spectrum, factor=5)        
        spectrum_envelopes.append(log_freq)
    return np.array(spectrum_envelopes).T

def get_wavelet_envelopes(frames, level=5, window_type='hann'):
    '''
    Decomposes input audio with wavelet packets and calculates energy envelopes of their components

    Parameters:
    frames          : overlapping signal frames for short-time analysis
    level           : number of levels of wavelet decomposition, 2**level gives the final number of wavelet components
    window_type     : type of windowing function to apply

    Returns numpy 2D array af 
    '''    

    frame_size = len(frames[0])
    num_bands = 2**level
    output_envelopes = {i: [] for i in range(num_bands)}
    windowing = es.Windowing(type='hann', size=frame_size)
    
    for frame in frames:
        wp = pywt.WaveletPacket(data=windowing(frame), wavelet='db1', mode='zero', maxlevel=level)
        for i in range(num_bands):            
            band_key = bin(i).replace('0b','').zfill(level).replace('0', 'a').replace('1','d')            
            output_envelopes[i].append(np.std(wp[band_key].data))
    
    output_array = []
    for item in output_envelopes.values():
        output_array.append(list(item))
    
    return np.array(output_array)

##########################################
#       Feature extractor class          #
##########################################

class feature_extractor():
    '''
    This class is used for reading audio data and metadata from aif. file,
    and for preprocessing: removing DC and normalizing perceptual loudness.
    '''
    
    def __init__(self, frames, sample_rate):
        '''
        Parameters: 
        frames          : overlapping signal frames for short-time analysis
        sample_rate     : audio sampling rate
        '''
        self.__frames = frames
        self.__samplerate = sample_rate

        self.__dataready = False  # is True after calling process() method
        self.__failed = False # is True if some exception occured while audio processing
        self.__framesize = len(frames[0])
        
        self.__ampspectra = None
        self.__melspectra = None
        self.__mfccs = None
        self.__lpcs = None
        self.__cqs = None
        self.__spes = None
        self.__waveletenvelopes = None

    @property
    def frames(self):
        return self.__frames
    
    @property
    def frame_size(self):
        return self.__framesize
    
    @property
    def sample_rate(self):
        return self.__samplerate
    
    def process(self, num_features=32, win_type='hann'):
        '''
        Calculates all features.
        Parameters:
        num_features : number of MFC coefficients, LP coefficients and wavelet decomposition bands, should be a power of 2
        win_type     : string alias for a windowing function
        '''
        assert(bin(num_features).count('1') == 1)
        try:
            self.__ampspectra, self.__melspectra, self.__mfccs = get_mfcc(
                self.__frames, sample_rate=self.__samplerate,
                num_bands=num_features*2,
                num_coeffs=num_features,
                window_type=win_type
            )
            self.__lpcs, _ = get_lpc(
                self.__frames,
                sample_rate=self.__samplerate,
                num_coeffs=num_features
            )
            self.__cqs = get_constantq(
                self.__frames,
                sample_rate=self.__samplerate,
                num_bands=num_features
            )
            self.__spes = get_spectrum_envelopes(
                self.__lpcs,
                sample_rate=self.__samplerate,
                num_bands=num_features
            )
            num_wavelet_levels = int(np.log2(num_features))
            self.__waveletenvelopes = get_wavelet_envelopes(
                self.__frames,
                level=num_wavelet_levels,
                window_type=win_type
            )
            self.__ready = True
            self.__failed = False
        except Exception as ex:
            print(ex)
            print('Processing failed!')
            self.__ampspectra = None
            self.__melspectra = None
            self.__mfccs = None
            self.__lpcs = None
            self.__cqs = None
            self.__spes = None
            self.__waveletenvelopes = None
            self.__failed = True
            self.__ready = False
    
    @property
    def amp_spectra(self):
        if not self.__ready:
            print('Call process() first!')
        return self.__ampspectra
    
    @property
    def mel_spectra(self):
        if not self.__ready:
            print('Call process() first!')
        return self.__melspectra

    @property
    def mfcc(self):
        if not self.__ready:
            print('Call process() first!')
        return self.__mfccs
    
    @property
    def lpc(self):
        if not self.__ready:
            print('Call process() first!')
        return self.__lpcs

    @property
    def cq(self):
        if not self.__ready:
            print('Call process() first!')
        return self.__cqs

    
    @property
    def spe(self):
        if not self.__ready:
            print('Call process() first!')
        return self.__spes
    
    @property
    def wp_envelopes(self):
        if not self.__ready:
            print('Call process() first!')
        return self.__waveletenvelopes
    
    


