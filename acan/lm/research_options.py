"""Opt-in controlled protocol; legacy projects keep their exact trainer."""
import math
from .trainer import validate_options as legacy_options

EXTRA=dict(protocol='controlled-v1',architecture='hybrid',graft_fraction=.5,distillation_weight=0.,distillation_temperature=2.,
           early_stop_patience=3,early_stop_min_delta=.01,validation_batches=8)


def validate_options(options):
    extra={key:options.get(key,value) for key,value in EXTRA.items()}
    if isinstance(extra['graft_fraction'],bool) or extra['graft_fraction'] not in (.5,1.):
        raise ValueError('Frazione ACAN ammessa: 0.5 oppure 1.0.')
    if extra['architecture'] not in ('hybrid','dense-control'):
        raise ValueError('Architettura non supportata.')
    if extra['protocol']!='controlled-v1':
        raise ValueError('Protocollo di training non supportato.')
    for key,lo,hi in [('distillation_weight',0.,1.),('distillation_temperature',.5,5.),('early_stop_min_delta',0.,1.)]:
        value=extra[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not lo<=value<=hi:
            raise ValueError('Parametro non valido: '+key)
    for key,lo,hi in [('early_stop_patience',0,20),('validation_batches',1,64)]:
        value=extra[key]
        if isinstance(value,bool) or not isinstance(value,int) or not lo<=value<=hi:
            raise ValueError('Parametro non valido: '+key)
    result=legacy_options({key:value for key,value in options.items() if key not in EXTRA})
    result.update(extra)
    return result
