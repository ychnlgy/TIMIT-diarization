from .mfcc import mfcc

from .. import collecting, toolkit

MFCC = "mfcc"
TIMIT_SAMPLERATE = 16000
WINLEN = 0.025
WINSTEP = 0.01
NUMCEP = 16
NFILT = 26
NFFT = 512
LOWFREQ = 0
HIGHFREQ = None
PREEMPH = 0.97
CEPLIFTER = 22
APPENDENERGY = True

def preprocess(savepath):
    data, test = toolkit.save.load(savepath)
    return _preprocess(data), _preprocess(test)

def _preprocess(obj):
    '''

    Input:
        obj as defined by the dict:
        
        {<subject_id>: {
            <sample_id>: {
                "PHN": str <path-to-phoneme-annotations>,
                "TXT": str <path-to-text-annotations>,
                "WAV": str <path-to-wav-audio>,
                "WRD": str <path-to-word-annotations>,
                "phn-data": list of (int start, int end, str phoneme)
                "wav-data": numpy array of uint16, wave audio file.
            }
        }

    Output:
        obj as defined by the dict:
        
        {<subject_id>: {
            <sample_id>: {
                "PHN": str <path-to-phoneme-annotations>,
                "TXT": str <path-to-text-annotations>,
                "WAV": str <path-to-wav-audio>,
                "WRD": str <path-to-word-annotations>,
                "phn-data": list of (int start, int end, str phoneme)
                "wav-data": numpy array of uint16, wave audio file.
                "mfcc": numpy array of shape (NUMCEP, *), transformation of wav-data.
            }
        }

    '''
    return collecting.traverse(obj, _preprocess_data)

def _preprocess_data(sample_data):
    sample_data[MFCC] = mfcc(
        signal = sample_data[collecting.WAV_DATA],
        samplerate = TIMIT_SAMPLERATE,
        winlen = WINLEN,
        winstep = WINSTEP,
        numcep = NUMCEP,
        nfilt = NFILT,
        nfft = NFFT,
        lowfreq = LOWFREQ,
        highfreq = HIGHFREQ,
        preemph = PREEMPH,
        ceplifter = CEPLIFTER,
        appendEnergy = APPENDENERGY
    )
