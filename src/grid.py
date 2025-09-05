import numpy as np
from dataclasses import dataclass

# KODOWANIE KOMÓREK (uint8):
# 0 = puste ".",  1 = ściana "#/X",  10..14 = cyfry "0..4"  (wartość = 10 + cyfra)
# Dzięki temu cyfrę rozpoznajemy szybko: cell >= 10

@dataclass(frozen=True)
class NpBoard:
    H: int                # liczba wierszy
    W: int                # liczba kolumn
    grid: np.ndarray      # (H,W) uint8, patrz kodowanie powyżej
    empty_mask: np.ndarray      # (H,W) bool, 0=niepuste, 1=puste "."
    wall_mask:  np.ndarray      # (H,W) bool, 1=ściana lub cyfra
    digit_mask: np.ndarray      # (H,W) bool, 1=cyfra
    digit_val:  np.ndarray      # (H,W) int8, 0..4 na cyfrach, 0 gdzie indziej
    hseg_id:    np.ndarray      # (H,W) int32, ID segmentu poziomego (między ścianami/cyframi)
    vseg_id:    np.ndarray      # (H,W) int32, ID segmentu pionowego
    hseg_len:   np.ndarray      # (n_hseg,) int32, długości segmentów (liczone na empty+bulb-space, tj. nie-ściany)
    vseg_len:   np.ndarray      # (n_vseg,) int32

def parse_ascii(lines):
    """lines: iterator/seq, 1. linia 'R C', potem R linii planszy w ASCII."""
    it = iter(lines)
    H, W = map(int, next(it).split())
    raw = [list(next(it).rstrip("\n")) for _ in range(H)]
    grid = np.zeros((H, W), dtype=np.uint8)
    for y in range(H):
        for x, ch in enumerate(raw[y]):
            if ch == '.':
                grid[y, x] = 0
            elif ch in ('#', 'X'):
                grid[y, x] = 1
            elif ch.isdigit():
                grid[y, x] = 10 + int(ch)
            else:
                raise ValueError(f"Nieznany znak '{ch}' w ({y},{x})")
    return build_npboard_from_grid(grid)

def build_npboard_from_grid(grid: np.ndarray) -> NpBoard:
    H, W = grid.shape
    digit_mask = grid >= 10
    wall_mask  = (grid == 1) | digit_mask
    empty_mask = grid == 0
    digit_val  = np.where(digit_mask, grid - 10, 0).astype(np.int8)

    # --- ID segmentów poziomych (między ścianami/cyframi) ---
    # Nowy segment, gdy zaczynamy w kolumnie 0 (nie-ściana) albo po lewej jest ściana.
    # błąd!!!  start_h = (~wall_mask) & ( (np.arange(W)==0)[None,:] | wall_mask[:, :-1] )
    # Nadajemy narastające ID w każdym wierszu:
    h_id = np.zeros((H, W), dtype=np.int32) - 1
    next_id = 0
    for y in range(H):
        run = -1
        for x in range(W):
            if wall_mask[y, x]:
                run = -1
                continue
            if (x == 0) or wall_mask[y, x-1]:
                run = next_id
                next_id += 1
            h_id[y, x] = run
    n_hseg = next_id

    # Długości segmentów poziomych: zliczamy liczbę pól nie-ścian w danym segmencie
    valid_h = h_id >= 0
    hseg_len = np.bincount(h_id[valid_h], minlength=n_hseg).astype(np.int32)

    # --- ID segmentów pionowych ---
    v_id = np.zeros((H, W), dtype=np.int32) - 1
    next_id = 0
    for x in range(W):
        run = -1
        for y in range(H):
            if wall_mask[y, x]:
                run = -1
                continue
            if (y == 0) or wall_mask[y-1, x]:
                run = next_id
                next_id += 1
            v_id[y, x] = run
    n_vseg = next_id
    valid_v = v_id >= 0
    vseg_len = np.bincount(v_id[valid_v], minlength=n_vseg).astype(np.int32)

    return NpBoard(H=H, W=W, grid=grid,
                   empty_mask=empty_mask, wall_mask=wall_mask,
                   digit_mask=digit_mask, digit_val=digit_val,
                   hseg_id=h_id, vseg_id=v_id,
                   hseg_len=hseg_len, vseg_len=vseg_len)


# bulbs_mask: (H,W) bool — True tam, gdzie stoi żarówka.
# Dla metaheurystyk łatwo aktualizować „punktowo” (flip komórki) i policzyć delty.

def penalties(board: NpBoard, bulbs_mask: np.ndarray):
    H, W = board.H, board.W
    assert bulbs_mask.shape == (H, W)

    # Żarówki tylko na pustych polach
    if (bulbs_mask & ~board.empty_mask).any():
        # hard fail (opcjonalnie można karać zamiast przerywać)
        raise ValueError("Żarówki mogą stać tylko na pustych polach '.'")

    # --- Zliczanie żarówek w segmentach ---
    bulbs_idx = np.where(bulbs_mask)
    h_counts = np.bincount(board.hseg_id[bulbs_idx], minlength=board.hseg_len.size)
    v_counts = np.bincount(board.vseg_id[bulbs_idx], minlength=board.vseg_len.size)

    # --- Konflikty: w każdym segmencie C(n,2) ---
    def nC2(n): return n*(n-1)//2
    conflicts = nC2(h_counts).sum() + nC2(v_counts).sum()

    # --- Oświetlenie: komórka jest oświetlona, jeśli w jej hseg lub vseg jest >=1 żarówka
    any_h = h_counts > 0
    any_v = v_counts > 0
    lit = np.zeros((H, W), dtype=bool)
    valid = ~board.wall_mask
    lit[valid] = any_h[board.hseg_id[valid]] | any_v[board.vseg_id[valid]]

    # --- Liczba nieoświetlonych pustych pól ---
    unlit = np.count_nonzero(board.empty_mask & ~lit)

    # --- Cyfry: suma żarówek w 4 sąsiadach ---
    # Przesunięcia maski żarówek (góra/dół/lewo/prawo) i maskujemy do sąsiadów cyfr
    B = bulbs_mask
    up    = np.zeros_like(B);  up[1: , :] = B[:-1, :]
    down  = np.zeros_like(B);  down[:-1,:] = B[1:  , :]
    left  = np.zeros_like(B);  left[:,1: ] = B[:, :-1]
    right = np.zeros_like(B);  right[:,:-1] = B[:, 1: ]

    # Sąsiedzi muszą istnieć (brzegi) — maski powyżej już to uwzględniają.
    # Teraz liczymy tylko tam, gdzie stoi cyfra:
    neigh_bulbs = up.astype(np.int8) + down.astype(np.int8) + left.astype(np.int8) + right.astype(np.int8)
    diffs = np.abs(neigh_bulbs - board.digit_val)
    digit_errors = diffs[board.digit_mask].sum()

    return int(conflicts), int(digit_errors), int(unlit)

def auto_weights(board: NpBoard):
    # U: maks. liczba nieoświetlonych (wszystkie puste ciemne)
    U = int(board.empty_mask.sum())

    # D: maks. błąd cyfr — dla każdej cyfry bierzemy max(value, deg-value),
    # gdzie deg = liczba sąsiadów będących pustymi polami (0..4)
    # Wyznaczamy deg wektorowo:
    E = board.empty_mask
    upE    = np.zeros_like(E);  upE[1: , :] = E[:-1, :]
    downE  = np.zeros_like(E);  downE[:-1,:] = E[1:  , :]
    leftE  = np.zeros_like(E);  leftE[:,1: ] = E[:, :-1]
    rightE = np.zeros_like(E);  rightE[:,:-1] = E[:, 1: ]
    deg = (upE + downE + leftE + rightE).astype(np.int8)
    deg_d = deg[board.digit_mask]
    val_d = board.digit_val[board.digit_mask]
    D = int(np.maximum(val_d, np.maximum(0, deg_d - val_d)).sum())

    # C: górne ograniczenie konfliktów — suma C(len,2) po wszystkich segmentach h i v
    def nC2_vec(x): x = x.astype(np.int64); return (x*(x-1))//2
    C = int(nC2_vec(board.hseg_len).sum() + nC2_vec(board.vseg_len).sum())

    # w_unlit = 1
    # w_digit > U
    # w_conflict > D*w_digit + U
    wU = 1
    wD = U + 1
    wC = D * wD + U + 1
    # mikro-uelastycznienie: jeśli C==0 (brak segmentów dłuższych niż 1), zredukuj rangę konfliktów
    if C == 0:
        wC = wD + wU + 1
    return int(wC), int(wD), int(wU)

def objective_auto(board: NpBoard, bulbs_mask: np.ndarray) -> int:
    c, d, u = penalties(board, bulbs_mask)
    wC, wD, wU = auto_weights(board)
    return wC*c + wD*d + wU*u

import numpy as np

class IncrementalObjective:
    """
    Inkrementalna ocena: przechowuje liczniki w segmentach, maskę światła i licznik sąsiadów cyfr.
    Pozwala policzyć delta-score dla flipu żarówki (y,x) w O(długość hseg + długość vseg).
    """
    def __init__(self, board: NpBoard, bulbs_mask: np.ndarray):
        self.b = board
        H, W = board.H, board.W
        self.B = bulbs_mask.copy().astype(bool)

        # liczniki żarówek w segmentach
        bulbs_idx = np.where(self.B)
        self.h_counts = np.bincount(board.hseg_id[bulbs_idx], minlength=board.hseg_len.size)
        self.v_counts = np.bincount(board.vseg_id[bulbs_idx], minlength=board.vseg_len.size)

        # światło: pole widoczne, jeśli hseg>0 lub vseg>0
        any_h = self.h_counts > 0
        any_v = self.v_counts > 0
        self.lit = np.zeros((H, W), dtype=bool)
        valid = ~board.wall_mask
        self.lit[valid] = any_h[board.hseg_id[valid]] | any_v[board.vseg_id[valid]]

        # sąsiedzi cyfr (ile żarówek dotyka każdej cyfry)
        B = self.B
        up    = np.zeros_like(B);  up[1: , :] = B[:-1, :]
        down  = np.zeros_like(B);  down[:-1,:] = B[1:  , :]
        left  = np.zeros_like(B);  left[:,1: ] = B[:, :-1]
        right = np.zeros_like(B);  right[:,:-1] = B[:, 1: ]
        self.digit_neigh_bulbs = (up + down + left + right).astype(np.int8)

        # podstawowe składowe kary
        self.conflicts = int(((self.h_counts*(self.h_counts-1))//2).sum()
                           + ((self.v_counts*(self.v_counts-1))//2).sum())
        self.unlit = int(np.count_nonzero(board.empty_mask & ~self.lit))
        diffs = np.abs(self.digit_neigh_bulbs - board.digit_val)
        self.digit_errors = int(diffs[board.digit_mask].sum())

        # wagi
        self.wC, self.wD, self.wU = auto_weights(board)

    def score(self) -> int:
        return self.wC*self.conflicts + self.wD*self.digit_errors + self.wU*self.unlit

    def flip_delta(self, y: int, x: int) -> int:
        """Zwraca delta score dla flipu (bez stosowania). Jeśli pole niepuste -> +∞ (lub rzuć)."""
        if not self.b.empty_mask[y, x]:
            return +10**12  # bardzo zła zmiana

        add = not self.B[y, x]  # True jeśli dodajemy żarówkę
        s = 0

        # --- segmenty
        hid = self.b.hseg_id[y, x]
        vid = self.b.vseg_id[y, x]

        # konflikty: różnica C(n±1,2) - C(n,2) = ±n
        if add:
            s += self.wC * ( self.h_counts[hid] + self.v_counts[vid] )
        else:
            s -= self.wC * ( (self.h_counts[hid]-1) + (self.v_counts[vid]-1) )

        # --- unlit: pola w obu segmentach z/bez światła
        # pola h-segmentu
        mask_h = (self.b.hseg_id == hid) & (~self.b.wall_mask)
        lit_h_before = self.h_counts[hid] > 0
        # pola v-segmentu
        mask_v = (self.b.vseg_id == vid) & (~self.b.wall_mask)
        lit_v_before = self.v_counts[vid] > 0

        if add:
            # pola, które staną się oświetlone NOWO (tam gdzie wcześniej !(lit_h || lit_v))
            newly_lit = ( (~(lit_h_before | lit_v_before)) & (mask_h | mask_v) )
            s -= self.wU * int(np.count_nonzero(newly_lit & self.b.empty_mask))
        else:
            # jeśli to była JEDYNA żarówka w segmencie, pola mogą ściemnieć
            newly_unlit = np.zeros((), dtype=bool)
            cnt = 0
            if self.h_counts[hid] == 1:
                # ściemnieją pola zależne tylko od h, a v nie ma żarówki
                no_v = ~(self.v_counts[self.b.vseg_id] > 0)
                newly_unlit = mask_h & no_v
                cnt += int(np.count_nonzero(newly_unlit & self.b.empty_mask))
            if self.v_counts[vid] == 1:
                no_h = ~(self.h_counts[self.b.hseg_id] > 0)
                nu2 = mask_v & no_h
                cnt += int(np.count_nonzero(nu2 & self.b.empty_mask))
            s += self.wU * cnt

        # --- cyfry: 4 sąsiedzi wokół (y,x) mogą zmienić licznik
        diffs_before = 0
        diffs_after  = 0
        for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
            ny, nx = y+dy, x+dx
            if 0 <= ny < self.b.H and 0 <= nx < self.b.W and self.b.digit_mask[ny, nx]:
                before = abs(int(self.digit_neigh_bulbs[ny, nx]) - int(self.b.digit_val[ny, nx]))
                cur = int(self.digit_neigh_bulbs[ny, nx])
                after_cnt = cur + (1 if add else -1)
                after = abs(after_cnt - int(self.b.digit_val[ny, nx]))
                diffs_before += before
                diffs_after  += after
        s += self.wD * (diffs_after - diffs_before)

        return int(s)

    def apply_flip(self, y: int, x: int):
        """Wykonuje flip i aktualizuje wszystkie liczniki (zgodnie z flip_delta)."""
        delta = self.flip_delta(y, x)
        if delta >= 10**12:
            return delta  # niepuste pole – ignorujemy

        add = not self.B[y, x]
        self.B[y, x] = add

        # segmenty
        hid = self.b.hseg_id[y, x]
        vid = self.b.vseg_id[y, x]

        # aktualizacja konfliktów
        if add:
            self.conflicts += self.h_counts[hid] + self.v_counts[vid]
            self.h_counts[hid] += 1
            self.v_counts[vid] += 1
        else:
            self.conflicts -= (self.h_counts[hid]-1) + (self.v_counts[vid]-1)
            self.h_counts[hid] -= 1
            self.v_counts[vid] -= 1

        # aktualizacja unlit/lit
        any_h = self.h_counts > 0
        any_v = self.v_counts > 0
        valid = ~self.b.wall_mask
        new_lit = np.zeros_like(self.lit)
        new_lit[valid] = any_h[self.b.hseg_id[valid]] | any_v[self.b.vseg_id[valid]]

        before = int(np.count_nonzero(self.b.empty_mask & ~self.lit))
        after  = int(np.count_nonzero(self.b.empty_mask & ~new_lit))
        self.unlit = after
        self.lit = new_lit

        # aktualizacja cyfrowa (sąsiedzi 4)
        for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
            ny, nx = y+dy, x+dx
            if 0 <= ny < self.b.H and 0 <= nx < self.b.W and self.b.digit_mask[ny, nx]:
                before = abs(int(self.digit_neigh_bulbs[ny, nx]) - int(self.b.digit_val[ny, nx]))
                self.digit_neigh_bulbs[ny, nx] += (1 if add else -1)
                after = abs(int(self.digit_neigh_bulbs[ny, nx]) - int(self.b.digit_val[ny, nx]))
                self.digit_errors += (after - before)

        return int(delta)

