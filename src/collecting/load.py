import sphfile, csv, tqdm, sys

from . import collect

PHN_DATA = "phn-data"
WAV_DATA = "wav-data"

def load(datadir):
    data, test = collect.collect(datadir)
    return _load(data), _load(test)

def traverse(obj, fn):
    items = sorted(list(obj.items()))
    N = len(items)
    for i, (subject_id, subject_data) in enumerate(items, 1):
        sys.stderr.write("Subject %s (%d/%d)\n" % (subject_id, i, N))
        for sample_data in tqdm.tqdm(subject_data.values(), ncols=80, leave=False):
            fn(sample_data)
    return obj

def _load(collected):
    '''

    Input:
        collected as defined by the dict:
        
        {<subject_id>: {
            <sample_id>: {
                "PHN": str <path-to-phoneme-annotations>,
                "TXT": str <path-to-text-annotations>,
                "WAV": str <path-to-wav-audio>,
                "WRD": str <path-to-word-annotations>
            }
        }

    Output:
        collected as defined by the dict:
        
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
    return traverse(collected, _load_data)

def _load_data(sample_data):
    sample_data[PHN_DATA] = _load_phn(sample_data[collect.PHN])
    sample_data[WAV_DATA] = _load_wav(sample_data[collect.WAV])

def _load_phn(fpath):
    return list(_iter_phn(fpath))

def _iter_phn(fpath):
    with open(fpath, "r") as f:
        for i, j, phn in csv.reader(f, delimiter=" "):
            yield int(i), int(j), phn

def _load_wav(fpath):
    return sphfile.SPHFile(fpath).content
