from __future__ import annotations
from typing import Iterator, List, Optional, Set, Tuple, Callable, Any
import random
import numpy as np

Coord = Tuple[int, int]  # (y, x)


# ──────────────────────────────────────────────────────────────────────────────
# Abstrakcja planszy: działa z NpBoard (z grid.py) i z List[str]
# ──────────────────────────────────────────────────────────────────────────────

def _get_size(board: Any) -> Tuple[int, int]:
    if hasattr(board, "H") and hasattr(board, "W"):  # NpBoard
        return int(board.H), int(board.W)
    return len(board), len(board[0])  # List[str]

def _in_bounds(board: Any, y: int, x: int) -> bool:
    H, W = _get_size(board)
    return 0 <= y < H and 0 <= x < W

def _is_wall(board: Any, y: int, x: int) -> bool:
    if hasattr(board, "wall_mask"):        # NpBoard
        return bool(board.wall_mask[y, x])
    return board[y][x] == '#'              # List[str]

def _is_digit(board: Any, y: int, x: int) -> bool:
    if hasattr(board, "digit_mask"):       # NpBoard
        return bool(board.digit_mask[y, x])
    return board[y][x].isdigit()           # List[str]

def _is_empty(board: Any, y: int, x: int) -> bool:
    if hasattr(board, "empty_mask"):       # NpBoard
        return bool(board.empty_mask[y, x])
    return board[y][x] == '.'              # List[str]

def neighbors4(y: int, x: int) -> List[Coord]:
    return [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]

def _free_adjacent_empty(board: Any, bulbs: Set[Coord], y: int, x: int) -> List[Coord]:
    out: List[Coord] = []
    for ny, nx in neighbors4(y, x):
        if _in_bounds(board, ny, nx) and _is_empty(board, ny, nx) and (ny, nx) not in bulbs:
            out.append((ny, nx))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Miękko/twardo: czy wolno postawić żarówkę
# ──────────────────────────────────────────────────────────────────────────────

def _is_placeable_soft(board: Any, y: int, x: int) -> bool:
    # Dopuszczamy stany niepoprawne (kolizje bulb-bulb oceni funkcja celu)
    return _in_bounds(board, y, x) and _is_empty(board, y, x)

def _is_placeable_strict(board: Any, bulbs: Set[Coord], y: int, x: int) -> bool:
    # Zakaz ścian/cyfr + brak żarówki "w linii wzroku" do najbliższej ściany
    if not _is_placeable_soft(board, y, x):
        return False
    H, W = _get_size(board)

    # góra
    ty = y - 1
    while ty >= 0 and not _is_wall(board, ty, x):
        if (ty, x) in bulbs:
            return False
        ty -= 1
    # dół
    ty = y + 1
    while ty < H and not _is_wall(board, ty, x):
        if (ty, x) in bulbs:
            return False
        ty += 1
    # lewo
    tx = x - 1
    while tx >= 0 and not _is_wall(board, y, tx):
        if (y, tx) in bulbs:
            return False
        tx -= 1
    # prawo
    tx = x + 1
    while tx < W and not _is_wall(board, y, tx):
        if (y, tx) in bulbs:
            return False
        tx += 1
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Ruchy elementarne
# ──────────────────────────────────────────────────────────────────────────────

def op_add_random(board: Any, bulbs: Set[Coord], rng: random.Random,
                  strict: bool = False) -> Optional[Set[Coord]]:
    H, W = _get_size(board)
    candidates: List[Coord] = []
    for y in range(H):
        for x in range(W):
            if (y, x) in bulbs:
                continue
            ok = _is_placeable_strict(board, bulbs, y, x) if strict else _is_placeable_soft(board, y, x)
            if ok:
                candidates.append((y, x))
    if not candidates:
        return None
    y, x = rng.choice(candidates)
    new_bulbs = set(bulbs)
    new_bulbs.add((y, x))
    return new_bulbs

def op_remove_random(board: Any, bulbs: Set[Coord], rng: random.Random) -> Optional[Set[Coord]]:
    if not bulbs:
        return None
    rem = rng.choice(tuple(bulbs))
    new_bulbs = set(bulbs)
    new_bulbs.remove(rem)
    return new_bulbs

def op_move_random(board: Any, bulbs: Set[Coord], rng: random.Random,
                   radius: int = 1, strict: bool = False) -> Optional[Set[Coord]]:
    if not bulbs:
        return None
    by, bx = rng.choice(tuple(bulbs))
    candidates: List[Coord] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if abs(dy) + abs(dx) == 0 or abs(dy) + abs(dx) > radius:
                continue
            ny, nx = by + dy, bx + dx
            if not _in_bounds(board, ny, nx) or (ny, nx) in bulbs:
                continue
            ok = _is_placeable_strict(board, bulbs - {(by, bx)}, ny, nx) if strict else _is_placeable_soft(board, ny, nx)
            if ok:
                candidates.append((ny, nx))
    if not candidates:
        return None
    ny, nx = rng.choice(candidates)
    new_bulbs = set(bulbs)
    new_bulbs.remove((by, bx))
    new_bulbs.add((ny, nx))
    return new_bulbs

def op_toggle_random(board: Any, bulbs: Set[Coord], rng: random.Random,
                     strict: bool = False) -> Optional[Set[Coord]]:
    H, W = _get_size(board)
    pool: Set[Coord] = set(bulbs)
    for y in range(H):
        for x in range(W):
            if (y, x) in bulbs:
                continue
            ok = _is_placeable_strict(board, bulbs, y, x) if strict else _is_placeable_soft(board, y, x)
            if ok:
                pool.add((y, x))
    if not pool:
        return None
    y, x = rng.choice(tuple(pool))
    new_bulbs = set(bulbs)
    if (y, x) in new_bulbs:
        new_bulbs.remove((y, x))
    else:
        new_bulbs.add((y, x))
    return new_bulbs


# ──────────────────────────────────────────────────────────────────────────────
# “Lokalne naprawy” przy cyfrach – szybkie domykanie ograniczeń
# ──────────────────────────────────────────────────────────────────────────────

def _digit_need(board: Any, bulbs: Set[Coord], y: int, x: int) -> Tuple[int, int]:
    """Zwraca (need, have) dla cyfry na (y,x)."""
    if hasattr(board, "digit_val"):
        target = int(board.digit_val[y, x])      # NpBoard
    else:
        target = int(board[y][x])                # List[str]
    adj = neighbors4(y, x)
    have = sum((ny, nx) in bulbs for ny, nx in adj if _in_bounds(board, ny, nx))
    need = target - have
    return need, have

def op_digit_repair_add(board: Any, bulbs: Set[Coord], rng: random.Random) -> Optional[Set[Coord]]:
    H, W = _get_size(board)
    candidates: List[Tuple[Coord, List[Coord]]] = []
    for y in range(H):
        for x in range(W):
            if _is_digit(board, y, x):
                need, _ = _digit_need(board, bulbs, y, x)
                if need > 0:
                    frees = _free_adjacent_empty(board, bulbs, y, x)
                    if frees:
                        candidates.append(((y, x), frees))
    if not candidates:
        return None
    _, frees = rng.choice(candidates)
    ny, nx = rng.choice(frees)
    new_bulbs = set(bulbs)
    new_bulbs.add((ny, nx))
    return new_bulbs

def op_digit_repair_remove(board: Any, bulbs: Set[Coord], rng: random.Random) -> Optional[Set[Coord]]:
    H, W = _get_size(board)
    victims: List[Coord] = []
    for y in range(H):
        for x in range(W):
            if _is_digit(board, y, x):
                need, _ = _digit_need(board, bulbs, y, x)
                if need < 0:  # za dużo żarówek obok cyfry
                    for ny, nx in neighbors4(y, x):
                        if _in_bounds(board, ny, nx) and (ny, nx) in bulbs:
                            victims.append((ny, nx))
    if not victims:
        return None
    rem = rng.choice(victims)
    new_bulbs = set(bulbs)
    new_bulbs.remove(rem)
    return new_bulbs


# ──────────────────────────────────────────────────────────────────────────────
# Interfejs 1: enumerator – wiele sąsiadów (HC / Tabu)
# ──────────────────────────────────────────────────────────────────────────────

def enumerate_neighbors(board: Any,
                        bulbs: Set[Coord],
                        max_neighbors: int = 50,
                        rng: Optional[random.Random] = None,
                        use_strict: bool = False,
                        with_repairs: bool = True) -> Iterator[Set[Coord]]:
    """
    Generator do max_neighbors kandydatów, miksuje różne operatory.
    use_strict=True zakazuje konfliktów bulb-bulb już na etapie generowania.
    """
    if rng is None:
        rng = random.Random()

    ops: List[Callable[[], Optional[Set[Coord]]]] = [
        lambda: op_add_random(board, bulbs, rng, strict=use_strict),
        lambda: op_remove_random(board, bulbs, rng),
        lambda: op_move_random(board, bulbs, rng, radius=1, strict=use_strict),
        lambda: op_toggle_random(board, bulbs, rng, strict=use_strict),
    ]
    if with_repairs:
        ops += [
            lambda: op_digit_repair_add(board, bulbs, rng),
            lambda: op_digit_repair_remove(board, bulbs, rng),
        ]

    produced = 0
    while produced < max_neighbors:
        rng.shuffle(ops)
        for op in ops:
            if produced >= max_neighbors:
                break
            cand = op()
            if cand is not None and cand != bulbs:
                produced += 1
                yield cand


# ──────────────────────────────────────────────────────────────────────────────
# Interfejs 2: proposer – jeden sąsiad (SA)
# ──────────────────────────────────────────────────────────────────────────────

def propose_neighbor(board: Any,
                     bulbs: Set[Coord],
                     rng: Optional[random.Random] = None,
                     temperature: float = 1.0,
                     use_strict: bool = False,
                     with_repairs: bool = True) -> Set[Coord]:
    """
    Zwraca pojedynczego sąsiada; wagi operatorów lekko zależą od 'temperature'.
    """
    if rng is None:
        rng = random.Random()

    pool: List[Tuple[float, Callable[[], Optional[Set[Coord]]]]] = [
        (0.30 + 0.20*min(1.0, temperature), lambda: op_add_random(board, bulbs, rng, strict=use_strict)),
        (0.25, lambda: op_remove_random(board, bulbs, rng)),
        (0.30 + 0.20*min(1.0, temperature), lambda: op_move_random(board, bulbs, rng, radius=1, strict=use_strict)),
        (0.15, lambda: op_toggle_random(board, bulbs, rng, strict=use_strict)),
    ]
    if with_repairs:
        pool += [
            (0.25, lambda: op_digit_repair_add(board, bulbs, rng)),
            (0.20, lambda: op_digit_repair_remove(board, bulbs, rng)),
        ]

    total_w = sum(w for w, _ in pool)

    for _ in range(16):
        r = rng.random() * total_w
        acc = 0.0
        for w, op in pool:
            acc += w
            if r <= acc:
                cand = op()
                if cand is not None and cand != bulbs:
                    return cand
                break
    return set(bulbs)  # awaryjnie: brak zmiany


# ──────────────────────────────────────────────────────────────────────────────
# Helper: set[(y,x)] -> maska bool HxW (dla objective_auto)
# ──────────────────────────────────────────────────────────────────────────────

def bulbs_set_to_mask(board: Any, bulbs_set: Set[Coord]) -> np.ndarray:
    H, W = _get_size(board)
    mask = np.zeros((H, W), dtype=bool)
    for y, x in bulbs_set:
        if 0 <= y < H and 0 <= x < W:
            mask[y, x] = True
    return mask
