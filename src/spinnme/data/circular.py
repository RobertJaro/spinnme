"""CLI convenience API for the preserved spherical circular-pattern fixture."""
from pathlib import Path

from spinnme.configuration import write_config

from .synthetic import generate_circular


def create_circular_dataset(output, size=32, frames=2):
    """Generate FITS observations and a directly runnable small YAML config."""
    directory = Path(output).resolve()
    generate_circular(directory, shape=(size,size), n_time_steps=frames)
    dataset = {'type': 'fits', 'data_path': str(directory/'circular_???.fits'),
                   'axis_order': ['stokes','wavelength','y','x'], 'stokes_order': 'IQUV',
                   'stokes_normalization': 1., 'instrument_id': 'CIRCULAR', 'line_id': 'fe-i-6173'}
    config = {'path':'<<run_path>>', 'work_path':'<<work_path>>',
              'data':{'train_configs':[dataset], 'valid_config':{**dataset,'sample_idx':0},
                      'ref_time':'2025-01-01T00:00:00', 'num_workers':0,'num_dataset_workers':0,
                      'batch_size':128,'dataset_batch_size':64,'seconds_per_dt':3600,'gauss_per_dB':1000},
              'instrument':[{'type':'generic','instrument_id':'CIRCULAR','line_id':'fe-i-6173'}],
              'model':{'type':'mlp','dim':32,'n_layers':2,'vector_potential':False},
              'training':{'epochs':-1,'max_steps':5,'check_val_every_n_epoch':1,
                          'optimizer_config':{'type':'adamw'}},
              'logging':{'project':'spinnme','name':'circular','mode':'disabled'}}
    path = directory/'config.yaml'
    write_config(config, path)
    return path
