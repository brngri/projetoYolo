# -*- coding: utf-8 -*-
"""Carregamento e validação de assinaturas espectrais."""

import os
import json
import logging
from typing import List, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def load_signatures_from_json(
    json_paths: Union[str, List[str]]
) -> Tuple[np.ndarray, List[str]]:
    """
    Carrega assinaturas de um ou mais JSONs.
    Estrutura esperada: {"polygons": [{"id":..., "mean_spectrum":[...]}, ...]}
    Retorna (signatures [N, B], ids [N]).
    """
    if isinstance(json_paths, str):
        json_paths = [json_paths]

    all_signatures: List[np.ndarray] = []
    all_ids: List[str] = []

    for json_path in json_paths:
        if not os.path.isfile(json_path):
            logger.warning(f"JSON não encontrado, ignorando: {json_path}")
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'polygons' not in data:
            logger.warning(f"JSON sem 'polygons': {json_path}")
            continue

        for poly in data['polygons']:
            if 'mean_spectrum' not in poly or 'id' not in poly:
                continue
            all_signatures.append(np.asarray(poly['mean_spectrum'], dtype=np.float32))
            all_ids.append(str(poly['id']))

    if not all_signatures:
        raise ValueError("Nenhuma assinatura válida encontrada nos JSONs fornecidos")

    n_bands = len(all_signatures[0])
    for i, sig in enumerate(all_signatures):
        if len(sig) != n_bands:
            raise ValueError(
                f"Assinatura {i} tem {len(sig)} bandas, esperado {n_bands}"
            )

    signatures = np.vstack(all_signatures)
    logger.info(
        f"Carregadas {len(signatures)} assinaturas ({n_bands} bandas) "
        f"de {len(json_paths)} arquivo(s)"
    )
    return signatures, all_ids