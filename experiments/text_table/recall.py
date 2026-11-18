"""Ingressi riproducibili per il confronto del richiamo a distanza."""

import numpy as np


def recall_inputs(pool, filler, width, seed, trials, gaps):
    rng = np.random.default_rng(seed)
    # Draw ordinary observed tokens; reserve disjoint identifiers for cues and answers.
    pool = pool[pool >= 2]
    draw = rng.choice(pool, size=trials*(width+3)+width-1, replace=False)
    answers = draw[:2*trials]
    keys = draw[2*trials:3*trials]
    carrier = draw[3*trials:3*trials+width-1]
    random_cues = draw[3*trials+width-1:].reshape(trials, width)
    cues = np.vstack(([np.r_[carrier, key] for key in keys], random_cues))
    facts = np.column_stack((cues, answers)).ravel()
    filler = np.asarray(filler)
    filler = filler[~np.isin(filler, np.r_[answers, keys, random_cues[:, -1]])]
    if len(filler) < max(gaps):
        raise ValueError('Testo insufficiente dopo aver escluso chiavi e risposte.')
    prefix = np.r_[facts, filler[:max(gaps)]]
    return cues, answers, facts, prefix
