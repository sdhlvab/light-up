#!/usr/bin/env python3
from __future__ import annotations
import sys, argparse, json
from typing import Iterable

# --- Reprezentacja ---
# Używamy jednoliterowych symboli na wejściu/wyjściu, ale wewnętrznie mapujemy na kody liczbowe.
# Dzięki temu funkcja celu jest prosta i szybka (brak pracy na stringach).
# Możemy zostać przy tym backendzie i później bezboleśnie podmienić na NumPy (to tylko miejsce mapowania).

# Kody:
EMPTY = 0
BULB  = 1
WALL  = 2           # czarna bez cyfry (X)
NUM0  = 10          # 10..14 odpowiadają 0..4
NUM4  = 14

CHAR2ID = {
    '.': EMPTY,
    'B': BULB, '*': BULB,
    'X': WALL, '#': WALL,  # alternatywne
    '0': NUM0+0, '1': NUM0+1, '2': NUM0+2, '3': NUM0+3, '4': NUM0+4,
}

ID2CHAR = {v: k for k, v in {
    **{'.': EMPTY, 'B': BULB, 'X': WALL},
    **{'0': NUM0+0, '1': NUM0+1, '2': NUM0+2, '3': NUM0+3, '4': NUM0+4},
}.items()}

type Grid = list[list[int]]  # backend: list[list[int]] (łatwy do podmiany na NumPy później)


# --- I/O ---

def parse_ascii(lines: Iterable[str]) -> Grid:
    """Czyta planszę z linii tekstu -> Grid (kody liczbowe). Ignoruje puste linie i spacje."""
    grid: Grid = []
    for raw in lines:
        line = raw.rstrip('\n')
        if not line.strip():
            continue
        row: list[int] = []
        for ch in line:
            if ch == ' ':
                continue
            if ch not in CHAR2ID:
                raise ValueError(f"Nieznany znak w wejściu: {ch!r}. Dozwolone: . X 0..4 B/*")
            row.append(CHAR2ID[ch])
        grid.append(row)
    if not grid:
        raise ValueError("Pusta plansza na wejściu.")
    # weryfikacja prostokątności
    w = len(grid[0])
    if any(len(r) != w for r in grid):
        raise ValueError("Wiersze mają różną długość (plansza musi być prostokątna).")
    return grid

def grid_to_ascii(grid: Grid) -> list[str]:
    out: list[str] = []
    for r in grid:
        out.append("".join(ID2CHAR.get(v, '?') for v in r))
    return out


# --- Funkcje pomocnicze do oceny ---

# Czy komórka (i,j) jest „barierą” dla światła (ściana/cyfra)?
def is_barrier(v: int) -> bool:
    return v == WALL or (NUM0 <= v <= NUM4)

# Zwraca ile żarówek sąsiaduje ortogonalnie z komórką (i,j)
def adjacent_bulbs(grid: Grid, i: int, j: int) -> int:
    h, w = len(grid), len(grid[0])
    cnt = 0
    if i > 0     and grid[i-1][j] == BULB: cnt += 1
    if i+1 < h   and grid[i+1][j] == BULB: cnt += 1
    if j > 0     and grid[i][j-1] == BULB: cnt += 1
    if j+1 < w   and grid[i][j+1] == BULB: cnt += 1
    return cnt

# Liczy konflikty żarówek w wierszach i kolumnach (para żarówek widzących się bez bariery).
def count_bulb_conflicts(grid: Grid) -> int:
    h, w = len(grid), len(grid[0])
    conflicts = 0

    # Wiersze
    for i in range(h):
        seen_bulb = False
        for j in range(w):
            v = grid[i][j]
            if is_barrier(v):
                seen_bulb = False
            elif v == BULB:
                if seen_bulb:
                    conflicts += 1  # para z poprzednią widoczną żarówką
                seen_bulb = True
            else:
                # puste lub inne – nic, „promień” leci dalej
                pass

    # Kolumny
    for j in range(w):
        seen_bulb = False
        for i in range(h):
            v = grid[i][j]
            if is_barrier(v):
                seen_bulb = False
            elif v == BULB:
                if seen_bulb:
                    conflicts += 1
                seen_bulb = True
            else:
                pass

    return conflicts

# Maska oświetlenia: które puste pola są oświetlone przez dowolną żarówkę
def lit_mask(grid: Grid) -> list[list[bool]]:
    h, w = len(grid), len(grid[0])
    lit = [[False]*w for _ in range(h)]

    # dla każdej żarówki „rozsyłamy” światło w 4 kierunkach, do bariery
    for i in range(h):
        for j in range(w):
            if grid[i][j] != BULB:
                continue
            lit[i][j] = True  # komórka z żarówką też jest oświetlona
            # lewo
            jj = j-1
            while jj >= 0 and not is_barrier(grid[i][jj]):
                lit[i][jj] = True
                jj -= 1
            # prawo
            jj = j+1
            while jj < w and not is_barrier(grid[i][jj]):
                lit[i][jj] = True
                jj += 1
            # góra
            ii = i-1
            while ii >= 0 and not is_barrier(grid[ii][j]):
                lit[ii][j] = True
                ii -= 1
            # dół
            ii = i+1
            while ii < h and not is_barrier(grid[ii][j]):
                lit[ii][j] = True
                ii += 1

    return lit


# --- Funkcja celu (loss) ---

def compute_loss(
    grid: Grid,
    w_unlit: int = 5,
    w_conflict: int = 10,
    w_number: int = 7,
    w_illegal: int = 0,
) -> dict:
    """
    Zwraca słownik z rozbiciem kar i sumą:
    {
      'unlit': int,
      'conflict_pairs': int,
      'number_mismatch': int,
      'illegal_bulbs': int,
      'total': int
    }
    """
    h, w = len(grid), len(grid[0])

    # 1) Konflikty żarówek (para w linii wzroku)
    conflict_pairs = count_bulb_conflicts(grid)

    # 2) Mismatch pod cyframi
    mismatch = 0
    illegal_bulbs = 0
    for i in range(h):
        for j in range(w):
            v = grid[i][j]
            if NUM0 <= v <= NUM4:
                need = v - NUM0
                got = adjacent_bulbs(grid, i, j)
                if got != need:
                    mismatch += abs(got - need)
            elif v == BULB:
                # (opcjonalne) jeśli ktoś wstawił BULB w ścianę/cyfrę, tu możemy to wykrywać,
                # ale ponieważ sprawdzamy to wyżej w parse, w praktyce zostawiamy tylko hook:
                pass

    # 3) Nieoświetlone puste pola
    lit = lit_mask(grid)
    unlit = 0
    for i in range(h):
        for j in range(w):
            if grid[i][j] == EMPTY and not lit[i][j]:
                unlit += 1

    # 4) (opcjonalnie) żarówki w miejscach niedozwolonych
    if w_illegal:
        for i in range(h):
            for j in range(w):
                v = grid[i][j]
                if v == BULB:
                    # jeżeli to ściana/cyfra – policz (u nas i tak nie wystąpi, ale zostawiamy)
                    pass

    total = (
        w_unlit    * unlit +
        w_conflict * conflict_pairs +
        w_number   * mismatch +
        w_illegal  * illegal_bulbs
    )
    return {
        "unlit": unlit,
        "conflict_pairs": conflict_pairs,
        "number_mismatch": mismatch,
        "illegal_bulbs": illegal_bulbs,
        "total": total,
    }


# --- CLI ---

def cmd_score(args: argparse.Namespace) -> int:
    # źródło danych: plik lub stdin
    if args.input and args.input != "-":
        with open(args.input, "r", encoding="utf-8") as f:
            grid = parse_ascii(f.readlines())
    else:
        grid = parse_ascii(sys.stdin.readlines())

    res = compute_loss(
        grid,
        w_unlit=args.w_unlit,
        w_conflict=args.w_conflict,
        w_number=args.w_number,
        w_illegal=args.w_illegal,
    )

    if args.json:
        print(json.dumps(res, ensure_ascii=False))
    else:
        print(f"TOTAL: {res['total']}")
        print(f"  unlit            = {res['unlit']}  (w={args.w_unlit})")
        print(f"  conflict_pairs   = {res['conflict_pairs']}  (w={args.w_conflict})")
        print(f"  number_mismatch  = {res['number_mismatch']}  (w={args.w_number})")
        if args.w_illegal:
            print(f"  illegal_bulbs    = {res['illegal_bulbs']}  (w={args.w_illegal})")
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lightup",
        description="Light Up (Akari) — narzędzia CLI: scoring / (wkrótce) neighbors / random",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # subcommand: score
    s = sub.add_parser("score", help="Policz funkcję celu (loss) dla podanej planszy/rozwiązania.")
    s.add_argument("input", nargs="?", default="-",
                   help="Plik z planszą (ASCII). Użyj '-' aby czytać ze stdin. Domyślnie: '-'")
    s.add_argument("--w-unlit", type=int, default=5, help="Waga kary za nieoświetlone pole (domyślnie 5)")
    s.add_argument("--w-conflict", type=int, default=10, help="Waga kary za parę żarówek w linii (domyślnie 10)")
    s.add_argument("--w-number", type=int, default=7, help="Waga kary za niedopasowanie do cyfry (domyślnie 7)")
    s.add_argument("--w-illegal", type=int, default=0, help="Waga kary za żarówkę w miejscu niedozwolonym (domyślnie 0)")
    s.add_argument("--json", action="store_true", help="Zwróć wynik w formacie JSON")
    s.set_defaults(func=cmd_score)

    # (kolejne subkomendy dodamy w następnych krokach)
    # neighbors  -> wygeneruje bliskie ruchy
    # random     -> wygeneruje losowe rozstawienie żarówek

    return p

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
