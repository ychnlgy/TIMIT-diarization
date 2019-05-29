import sphfile, csv

from . import collect

PHN_DATA = "phn-data"
WAV_DATA = "wav-data"

def load(datadir):
    data, test = collect.collect(datadir)
    return _load(data), _load(test)

def _load(collected):
    '''

    Input:
        dataset, testset as defined by the dict:
        
        {<subject_id>: {
            <sample_id>: {
                "PHN": str <path-to-phoneme-annotations>,
                "TXT": str <path-to-text-annotations>,
                "WAV": str <path-to-wav-audio>,
                "WRD": str <path-to-word-annotations>
            }
        }

    Output:
        dataset, testset as defined by the dict:
        
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

    '''
    for subject_data in collected.values():
        for sample_data in subject_data.values():
            sample_data[PHN_DATA] = _load_phn(sample_data[collect.PHN])
            sample_data[WAV_DATA] = _load_wav(sample_data[collect.WAV])
    return collected

def _load_phn(fpath):
    return list(_iter_phn(fpath))

def _iter_phn(fpath):
    with open(fpath, "r") as f:
        for i, j, phn in csv.reader(f, delimiter=" "):
            yield int(i), int(j), phn

def _load_wav(fpath):
    return sphfile.SPHFile(fpath).content
