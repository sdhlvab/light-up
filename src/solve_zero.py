from __future__ import annotations
from typing import Tuple, Set, Optional
import random
import numpy as np

from grid import NpBoard, objective_auto
from neighbors import propose_neighbor, bulbs_set_to_mask
from random_init import random_solution

Coord = Tuple[int, int]

def solve_zero(board: NpBoard,
               init_mode: str = "strict",
               fill_ratio: float = 0.08,
               seed: Optional[int] = None,
               max_iters: int = 200_000,
               temperature: float = 1.0,
               cooling: float = 0.9995,
               strict_neighbors: bool = False) -> Tuple[Set[Coord], int, int]:
    """
    Szuka rozwiązania o score=0.
    Zwraca: (bulbs_set, iterations_used, final_score).
    Pozycje żarówek nie są nigdzie drukowane – zwróć je wyżej i ew. zapisz do pliku.
    """
    rng = random.Random(seed)

    # Inicjalizacja
    bulbs_mask, bulbs = random_solution(board, mode=init_mode,
                                        fill_ratio=fill_ratio, seed=seed)
    score = objective_auto(board, bulbs_mask)

    it = 0
    T = temperature

    # Prosty SA z akceptacją gorszych ruchów, aż do score=0
    while it < max_iters and score > 0:
        it += 1
        cand = propose_neighbor(board, bulbs, rng=rng,
                                temperature=T,
                                use_strict=strict_neighbors,
                                with_repairs=True)
        if cand == bulbs:
            T *= cooling
            continue

        # oceń kandydata
        cand_mask = bulbs_set_to_mask(board, cand)
        cand_score = objective_auto(board, cand_mask)

        # akceptacja
        if cand_score <= score:
            bulbs, bulbs_mask, score = cand, cand_mask, cand_score
        else:
            # akceptacja probabilistyczna
            # uwaga: score to duże liczby, skaluje je T
            delta = cand_score - score
            # żeby uniknąć underflow: przy bardzo dużych delta/T prawdopodobieństwo ≈ 0
            from math import exp
            p = exp(-delta / max(1e-9, T))
            if rng.random() < p:
                bulbs, bulbs_mask, score = cand, cand_mask, cand_score

        # chłodzenie
        T *= cooling

    return bulbs, it, score
