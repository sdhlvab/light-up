from __future__ import annotations
from typing import Tuple, Set, Optional
import random
import numpy as np

from grid import NpBoard, objective_auto
from random_init import random_solution
from neighbors import bulbs_set_to_mask

Coord = Tuple[int, int]


def solve_random(board: NpBoard,
                 tries: int = 1000,
                 init_mode: str = "soft",
                 fill_ratio: float = 0.08,
                 seed: Optional[int] = None) -> Tuple[Set[Coord], int]:
    """
    Losowe przeszukiwanie:
    - generuje `tries` niezależnych rozwiązań,
    - ocenia każde funkcją celu,
    - zwraca najlepsze (z najniższym score).

    Zwraca: (bulbs_set, best_score)
    """
    rng = random.Random(seed)
    best_score = None
    best_bulbs: Set[Coord] = set()

    for _ in range(tries):
        mask, bulbs = random_solution(board,
                                      mode=init_mode,
                                      fill_ratio=fill_ratio,
                                      seed=rng.randint(0, 10 ** 9))
        score = objective_auto(board, mask)
        if best_score is None or score < best_score:
            best_score = score
            best_bulbs = bulbs

    return best_bulbs, int(best_score)


# ------------------------------------------------------------
# Test lokalny
# ------------------------------------------------------------
if __name__ == "__main__":
    from grid import parse_ascii
    import sys, json

    if len(sys.argv) < 2:
        print("Użycie: python solve_random.py plansza.txt [tries]", file=sys.stderr)
        sys.exit(1)

    board = parse_ascii(open(sys.argv[1]).read().splitlines())
    tries = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    bulbs, score = solve_random(board, tries=tries, init_mode="digit_greedy", fill_ratio=0.1)
    mask = bulbs_set_to_mask(board, bulbs)

    out = {
        "score": score,
        "bulbs_count": int(mask.sum()),
        "H": board.H, "W": board.W,
        "bulbs": sorted([[int(y), int(x)] for (y, x) in bulbs])
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
