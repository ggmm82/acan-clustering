"""Conservative CPU mechanics profile; reject oversized jobs before loading weights."""


def check_profile(options,cuda_available):
    if options.get('device')=='cuda' and not cuda_available:
        raise ValueError('CUDA non disponibile: usa il profilo CPU 135M oppure collega un worker GPU.')
    if options.get('device')!='cpu' and cuda_available:
        return
    if options['base_model']!='HuggingFaceTB/SmolLM2-135M':
        raise ValueError('Il profilo CPU verificato supporta SmolLM2-135M; modelli maggiori richiedono CUDA.')
    if (options['batch_size']!=1 or options['sequence_length']>128 or options['slots']>16
            or options['window']>16 or options['lora_rank']>8):
        raise ValueError('Profilo CPU conservativo: batch 1, blocchi fino a 128 token, massimo 16 slot/finestra e rank LoRA 8. Per budget maggiori usa CUDA.')
    if options.get('architecture','hybrid')=='hybrid' and options.get('graft_fraction',.5)==1. and options['sequence_length']>64:
        raise ValueError('Profilo CPU interamente ACAN: usa blocchi fino a 64 token.')
