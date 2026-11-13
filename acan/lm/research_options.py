"""Opt-in controlled protocol; legacy projects keep their exact trainer."""
import math
from .trainer import validate_options as legacy_options

EXTRA=dict(protocol='controlled-v1',routing_profile='shared-algebra-v1',architecture='hybrid',graft_fraction=.5,distillation_weight=0.,distillation_temperature=2.,
           early_stop_patience=3,early_stop_min_delta=.01,validation_batches=8,cp_weight=0.)


def validate_options(options):
    extra={key:options.get(key,value) for key,value in EXTRA.items()}
    if isinstance(extra['graft_fraction'],bool) or extra['graft_fraction'] not in (.5,1.):
        raise ValueError('Frazione ACAN ammessa: 0.5 oppure 1.0.')
    if extra['routing_profile'] not in ('legacy-causal-v1','shared-algebra-v1','cp-routing-v1','cp-joint-v1','cp-v9','concept-graph-v1'):
        raise ValueError('Profilo algebrico non valido.')
    if extra['architecture'] not in ('hybrid','dense-control'):
        raise ValueError('Architettura non supportata.')
    if extra['protocol']!='controlled-v1':
        raise ValueError('Protocollo di training non supportato.')
    for key,lo,hi in [('distillation_weight',0.,1.),('distillation_temperature',.5,5.),('early_stop_min_delta',0.,1.),('cp_weight',0.,1.)]:
        value=extra[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not lo<=value<=hi:
            raise ValueError('Parametro non valido: '+key)
    for key,lo,hi in [('early_stop_patience',0,20),('validation_batches',1,64)]:
        value=extra[key]
        if isinstance(value,bool) or not isinstance(value,int) or not lo<=value<=hi:
            raise ValueError('Parametro non valido: '+key)
    result=legacy_options({key:value for key,value in options.items() if key not in EXTRA})
    if extra['routing_profile'] in ('cp-routing-v1','cp-joint-v1') and result['device']!='cuda' and (result['sequence_length']>32 or result['slots']>8):
        raise ValueError('CP diretta su CPU: massimo 32 token per blocco e 8 slot. La GUI applica questo budget ai nuovi progetti.')
    result.update(extra)
    if extra['cp_weight'] and (extra['routing_profile'] not in ('cp-routing-v1','cp-joint-v1') or extra['architecture']!='hybrid'):
        raise ValueError('Il peso CP richiede il routing CP diretta e l’architettura ibrida.')
    return result
