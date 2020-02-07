import numpy as np
import scipy.signal as ssgn

import essentia.standard as es
import essentia

import librosa

import webrtc_vad as wvad

import aifc

from collections import namedtuple


##########################################
#           Utility functions            #
##########################################

def get_metadata(file_path):
    """    
    Takes the path to .aif file, and returns 2 tuples:
    audio sampling rate, number of channels, sample width, data size in samples,
    markers.
    """
    params = None
    markers = None
    try:
        with aifc.open(file_path, mode='rb') as aif_read:
            params = aif_read.getparams()
            markers = aif_read.getmarkers()
    except:
        print(f'Reading {file_path} failed.')
    return params, markers

def get_pcm_data(file_path):
    """
    Reads an .aif file.
    Takes the path, and returns PCM audio data as bytes.
    """
    pcm_data = None
    try:
        with aifc.open(file_path, mode='rb') as aif_read:
            pcm_data = aif_read.readframes(aif_read.getnframes())
    except:
        print(f'Reading {file_path} failed.')
    return pcm_data

def get_integer_data(file_path, padding=True, frame_size=512, hop_size=256):
    """
    Reads an .aif file.
    Takes a path, and returns integer audio data as numpy array.
    Zero padding is made so that the last frame is of the full size while short time framing
    """
    pcm_data = get_pcm_data(file_path)
    dt = np.dtype('int16')
    dt = dt.newbyteorder('>')
    integer_data = np.frombuffer(pcm_data, dtype=dt).astype('int16')
    if padding:
        actual_size = len(integer_data)
        number_of_frames = int(np.floor((actual_size - frame_size) / hop_size) + 1)
        padded_size = (number_of_frames + 1) * hop_size + frame_size
        padding_length = padded_size - actual_size
        integer_data = np.append(integer_data, np.zeros(padding_length, dtype='int16'))
    return integer_data

def get_float_data(file_path, sample_rate, padding=True, frame_size=512, hop_size=256):
    """
    Reads an .aif file.
    Takes the path, and returns float audio data in range [0, 1] as numpy array.
    Zero padding is made so that the last frame is of the full size while short time framing
    """    
    loader = es.MonoLoader(filename=file_path, sampleRate=sample_rate)
    float_data = loader()
    if padding:
        actual_size = len(float_data)
        number_of_frames = int(np.floor((actual_size - frame_size) / hop_size) + 1)
        padded_size = (number_of_frames + 1) * hop_size + frame_size
        padding_length = padded_size - actual_size
        float_data = np.append(float_data, np.zeros(padding_length, dtype='float32'))
    return float_data

def convert_to_pcm(numpy_data):
    '''
    Converts numpy 1D array to a bytes object
    '''
    int16_data = numpy_data.astype('int16')
    return int16_data.tobytes()

def get_voice_band(input_audio, sample_rate, filter_order=128, start_freq=300, stop_freq=4000):
    '''
    Applies band pass filter
    '''
    filter_coeffs = ssgn.firwin(filter_order, [start_freq, stop_freq], fs=sample_rate, pass_zero=False)
    filter_output = ssgn.convolve(input_audio, filter_coeffs, mode='valid')
    return filter_output

vadregion = namedtuple('vadregion',['voiced','start_idx', 'stop_idx', 'size'])

def get_voice_activity(integer_data, sample_rate=16000, sample_width=2, frame_size=512, hop_size=256, aggressiveness=1, do_filtering=True):
    '''
    Voice activite detection using WebRTC algorythm
    '''
    if do_filtering:
        filter_output = convert_to_pcm(get_voice_band(integer_data, sample_rate))
    else:
        filter_output = convert_to_pcm(integer_data)
    
    vad_flags, vad_regions = wvad.VAD_pcm(
        filter_output,
        sample_rate, sample_width,
        frame_size, hop_size,
        aggressiveness
    )
    
    return vad_flags, [vadregion._make(reg) for reg in vad_regions]

def markers_to_flags(markers, data_size, frame_size, hop_size):
    '''
    Converts aif markers to a list of flags and a list of vadregions
    '''
    first_frame = int(np.floor(markers[0][1] / hop_size))
    last_frame = int(np.ceil(markers[1][1] / hop_size))
    number_of_frames = int(np.floor((data_size - frame_size) / hop_size)) + 1
    presence_flags = []
    for i in range(number_of_frames):
        if first_frame <= i <= last_frame:
            presence_flags.append(True)
        else:
            presence_flags.append(False)
    presence_regions = []    
    presence_regions.append(vadregion._make([False, 0, first_frame - 1, first_frame - 1]))
    presence_regions.append(vadregion._make([True, first_frame, last_frame, last_frame - first_frame]))
    presence_regions.append(vadregion._make([False, last_frame + 1, number_of_frames - 1, number_of_frames - last_frame]))
    return presence_flags, presence_regions

def remove_dc(input_signal, sample_rate, cutoff_Hz=40):
    '''
    Remove 'direct current', i.e. lowest frequencies starting from zero up to a given cutoff frequency in Hertz.
    '''
    dc_remover = es.DCRemoval(cutoffFrequency=cutoff_Hz, sampleRate=sample_rate)
    return dc_remover(input_signal)

def apply_replay_gain(float_signal, sample_rate):
    '''
    Normalizes perceived loudness af an audio signal.
    Calculates a replay gain value and applies this gain to the input.
    
    Returns normalized signal and the replay gain calculated.
    '''
    downsampled_signal = es.Resample(inputSampleRate=sample_rate, outputSampleRate=8000)(float_signal)
    replay_gain_dB = es.ReplayGain(sampleRate=8000)(downsampled_signal)
    gain = np.power(10, replay_gain_dB / 20)
    return np.array(float_signal) * gain, replay_gain_dB

def get_informative_frames(input_data, markers, parameters, frame_size, hop_size):
    '''
    Takes as input audio data with its markers and parameters,
    and generates informative and noise frames according to markers, frame and hop sizes.
    Returns framed audio, duration of the informartive region, standard deviations of both informative and non-informative parts.
    '''
    first_informative_sample = markers[0][1]
    last_informative_sample = markers[1][1]
    noise_signal = np.append(input_data[0:first_informative_sample], input_data[last_informative_sample:])
    informative_signal = input_data[first_informative_sample : last_informative_sample]
    noise_rms = np.std(noise_signal)
    informative_rms = np.std(informative_signal)
    informative_duration = (last_informative_sample - first_informative_sample) / parameters.framerate
    first_informative_frame = int(np.floor(markers[0][1] / hop_size))
    last_informative_frame = int(np.ceil(markers[1][1] / hop_size))
    informative_frames = []
    noise_frames = []
    for frame_idx, frame in enumerate(es.FrameGenerator(input_data, frameSize=frame_size, hopSize=hop_size, startFromZero=True)):
        if first_informative_frame <= frame_idx <= last_informative_frame:
            informative_frames.append(frame)
        else:
            noise_frames.append(frame)
    return np.array(informative_frames), np.array(noise_frames), informative_duration, informative_rms, noise_rms


##########################################
#           Preprocessor class           #
##########################################

class aif_preprocessor():
    '''
    This class is used for reading audio data and metadata from aif. file,
    and for preprocessing: removing DC and normalizing perceptual loudness.
    '''
    
    def __init__(self, string_file_path, frame_size, hop_size):
        '''
        Parameters: 
        string_file_path (str)  : string containing path to the source .aif file
        frame_size (int)        : short-time analysis frame size in samples
        hop_size (int)          : short-time analysis step in samples between sequential frames
        '''
        self.__path = string_file_path
        self.__audioparams, self.__audiomarkers = get_metadata(self.__path)
        assert(self.__audioparams is not None)
        self.__dataready = False  # is True after calling process() method
        self.__failed = False # is True if some exception occured while audio processing
        self.__framesize = frame_size
        self.__hopsize = hop_size        
        assert(hop_size <= frame_size)
        
        self.__floatdata = None
        self.__integerdata = None
        self.__signalframes = None
        self.__noiseframes = None
        self.__signalduration = None
        self.__signalrms = None
        self.__noiserms = None        
        
    
    @property
    def audio_path(self):
        return self.__path
    
    @property
    def audio_info(self):
        return self.__audioparams
    
    @property
    def audio_markers(self):
        return self.__audiomarkers
    
    @property
    def frame_size(self):
        return self.__framesize
    
    @frame_size.setter
    def frame_size(self, new_value):
        self.__framesize = new_value
        self.__dataready = False
    
    @property
    def hop_size(self):
        return self.__hopsize
    
    @hop_size.setter
    def hop_size(self, new_value):
        self.__hopsize = new_value
        self.__dataready = False
        
    def process(self):
        try:
            tmp_data = get_float_data(
                self.__path,
                self.__audioparams.framerate,
                padding=True,
                frame_size=self.__framesize, hop_size=self.__hopsize
            )            
            tmp_data = remove_dc(tmp_data, sample_rate=self.__audioparams.framerate, cutoff_Hz=40)            
            self.__floatdata, self.__replaygain = apply_replay_gain(tmp_data, sample_rate=self.__audioparams.framerate)            
            self.__signalframes, self.__noiseframes,\
            self.__signalduration, self.__signalrms, self.__noiserms = get_informative_frames(
                self.__floatdata,
                self.__audiomarkers, self.__audioparams,
                self.__framesize, self.__hopsize
            )            
            self.__ready = True
            self.__failed = False
        except Exception as ex:
            print(ex)
            print('Processing failed!')
            self.__signalframes = None
            self.__noiseframes = None
            self.__signalduration = None
            self.__signalrms = None
            self.__noiserms = None
            self.__failed = True
            self.__ready = False
    
    @property
    def data_ready(self):
        return self.__ready
        
    @property
    def signal_rms(self):
        if not self.__ready:
            print('Call process() first!')
        return self.__signalrms

    @property
    def noise_rms(self):
        if not self.__ready:
            print('Call process() first!')
        return self.__noiserms
    
    @property
    def float_data(self):
        if not self.__ready:
            print('Call process() first!')
        return self.__floatdata
    
    @property
    def signal_duration(self):
        if not self.__ready:
            print('Call process() first!')
        return self.__signalduration
    
    @property
    def signal_frames(self):
        if not self.__ready:
            print('Call process() first!')
        return self.__signalframes
    
    @property
    def noise_frames(self):
        if not self.__ready:
            print('Call process() first!')
        return self.__noiseframes    