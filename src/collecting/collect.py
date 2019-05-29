import os, collections

ROOT = "TIMIT"
DATA = "TRAIN"
TEST = "TEST"

PHN = "PHN"
TXT = "TXT"
WAV = "WAV"
WRD = "WRD"

def collect(datadir):
    '''

    Input:
        datadir - str root of TIMIT dataset, such that

            <datadir>/TIMIT/TEST/DR1/FAKS0/SA1.PHN

        exists.

    '''

    datapath = os.path.join(ROOT, datadir, DATA)
    testpath = os.path.join(ROOT, datadir, TEST)
    
    assert os.path.isdir(datapath)
    assert os.path.isdir(testpath)

    return (
        _collect(datapath),
        _collect(testpath)
    )

def _collect(root):
    '''

    Input:
        datadir - str root directory of the TIMIT dataset.

    Output:
        dataset, testset as defined by the dict:
        
        {<subject_id>: {
            <sample_id>: {
                "PHN": str <path-to-phoneme-annotations>,
                "TXT": str <path-to-text-annotations>,
                "WAV": str <path-to-wav-audio>,
                "WRD": str <path-to-word-annotations>
            }
        }

    '''
    subject_data = {}
    
    for dialect in os.listdir(root):
        dialect_root = os.path.join(root, dialect)
        
        for subject in os.listdir(dialect_root):
            subject_root = os.path.join(dialect_root, subject)

            sample_data = collections.defaultdict(dict)
            for fname in os.listdir(subject_root):
                fpath = os.path.join(subject_root, fname)
                sample, info = fname.split(".")
                assert info in [PHN, TXT, WAV, WRD]
                
                sample_data[sample][info] = fpath

            # subjects are not expected to have multiple dialects
            assert subject not in subject_data  
            subject_data[subject] = sample_data

    return dialect_data

    
    
    
