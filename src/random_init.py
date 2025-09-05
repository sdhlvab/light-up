from __future__ import annotations
from typing import Tuple, Set, List, Optional
import random
import numpy as np
from grid import NpBoard

Coord = Tuple[int, int]  # (y, x)


# ──────────────────────────────────────────────────────────────────────────────
# Konwersje set <-> mask
# ──────────────────────────────────────────────────────────────────────────────

def bulbs_set_to_mask(board: NpBoard, bulbs_set: Set[Coord]) -> np.ndarray:
    mask = np.zeros((board.H, board.W), dtype=bool)
    for y, x in bulbs_set:
        if 0 <= y < board.H and 0 <= x < board.W:
            mask[y, x] = True
    return mask

def bulbs_mask_to_set(mask: np.ndarray) -> Set[Coord]:
    ys, xs = np.where(mask)
    return {(int(y), int(x)) for y, x in zip(ys, xs)}


# ──────────────────────────────────────────────────────────────────────────────
# Pomocnicze
# ──────────────────────────────────────────────────────────────────────────────

def _empty_coords(board: NpBoard) -> List[Coord]:
    ys, xs = np.where(board.empty_mask)
    return [(int(y), int(x)) for y, x in zip(ys, xs)]

def _no_conflict_if_added(board: NpBoard, bulbs: Set[Coord], y: int, x: int) -> bool:
    """Brak konfliktu 'w linii wzroku' <=> brak żarówki w tym samym h- lub v-segmencie."""
    hid = board.hseg_id[y, x]
    vid = board.vseg_id[y, x]
    # Jeśli to nie-puste pole, nie kładziemy
    if not board.empty_mask[y, x]:
        return False
    # Szybki test przez przegląd istniejących żarówek (wystarcza do inicjalizacji).
    for (by, bx) in bulbs:
        if board.hseg_id[by, bx] == hid or board.vseg_id[by, bx] == vid:
            return False
    return True

def _adjacent_empty(board: NpBoard, y: int, x: int) -> List[Coord]:
    out: List[Coord] = []
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < board.H and 0 <= nx < board.W and board.empty_mask[ny, nx]:
            out.append((ny, nx))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Główne generatory
# ──────────────────────────────────────────────────────────────────────────────

def random_bulbs_mask(board: NpBoard, p: float, seed: Optional[int] = None) -> np.ndarray:
    """
    'Soft' – po prostu losowo kładziemy żarówki na pustych polach z prawdopodobieństwem p.
    Może wprowadzać konflikty – to jest ok dla losowego startu do SA.
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros((board.H, board.W), dtype=bool)
    draws = rng.random((board.H, board.W))
    mask[board.empty_mask] = draws[board.empty_mask] < p
    return mask

def random_bulbs_mask_strict(board: NpBoard, p: float, seed: Optional[int] = None) -> np.ndarray:
    """
    'Strict' – zachowujemy brak konfliktów bulb–bulb w segmencie (h/v). Prosty zachłanny MIS.
    """
    rng = random.Random(seed)
    empties = _empty_coords(board)
    rng.shuffle(empties)

    bulbs: Set[Coord] = set()
    target = int(round(len(empties) * max(0.0, min(1.0, p))))
    for (y, x) in empties:
        if len(bulbs) >= target:
            break
        if _no_conflict_if_added(board, bulbs, y, x):
            bulbs.add((y, x))

    return bulbs_set_to_mask(board, bulbs)

def random_bulbs_mask_digit_greedy(board: NpBoard, p: float, seed: Optional[int] = None) -> np.ndarray:
    """
    'Digit-greedy' – najpierw próbujemy zaspokoić cyfry (bez konfliktów), potem dopychamy jak 'strict'.
    """
    rng = random.Random(seed)
    bulbs: Set[Coord] = set()

    # 1) Priorytet: cyfry
    digits: List[Tuple[int,int,int]] = []  # (priority, y, x) – tu priorytet = wartość cyfry (większe pierwsze)
    ys, xs = np.where(board.digit_mask)
    for y, x in zip(ys, xs):
        digits.append((int(board.digit_val[y, x]), int(y), int(x)))
    digits.sort(reverse=True)  # większe cyfry najpierw

    for _, y, x in digits:
        need = int(board.digit_val[y, x])
        adj = _adjacent_empty(board, y, x)
        rng.shuffle(adj)
        for (ny, nx) in adj:
            if need <= 0:
                break
            if _no_conflict_if_added(board, bulbs, ny, nx):
                bulbs.add((ny, nx))
                need -= 1
        # jeśli nie starczyło pustych sąsiadów, to trudno – to tylko inicjalizacja

    # 2) Dopychanie do celu jak 'strict'
    empties = _empty_coords(board)
    rng.shuffle(empties)
    target = int(round(board.empty_mask.sum() * max(0.0, min(1.0, p))))
    for (y, x) in empties:
        if len(bulbs) >= target:
            break
        if (y, x) in bulbs:
            continue
        if _no_conflict_if_added(board, bulbs, y, x):
            bulbs.add((y, x))

    return bulbs_set_to_mask(board, bulbs)

def random_solution(board: NpBoard,
                    mode: str = "soft",
                    fill_ratio: float = 0.08,
                    seed: Optional[int] = None) -> Tuple[np.ndarray, Set[Coord]]:
    """
    Wspólny interfejs:
      - mode ∈ {"soft", "strict", "digit_greedy"}
      - fill_ratio ∈ [0,1] – docelowy udział żarówek na pustych polach
    Zwraca: (bulbs_mask, bulbs_set)
    """
    mode = mode.lower()
    if mode == "soft":
        mask = random_bulbs_mask(board, fill_ratio, seed=seed)
    elif mode == "strict":
        mask = random_bulbs_mask_strict(board, fill_ratio, seed=seed)
    elif mode in ("digit", "digit_greedy", "digit-greedy"):
        mask = random_bulbs_mask_digit_greedy(board, fill_ratio, seed=seed)
    else:
        raise ValueError(f"Unknown init mode: {mode}")

    return mask, bulbs_mask_to_set(mask)
