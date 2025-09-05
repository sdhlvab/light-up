#!/usr/bin/env python3
import argparse
import sys
import json
import random
import numpy as np

from grid import parse_ascii, objective_auto, penalties, auto_weights, NpBoard
from neighbors import enumerate_neighbors, bulbs_set_to_mask
from random_init import random_bulbs_mask, random_solution

# ------------------------------------------------------------
# Wejście planszy
# ------------------------------------------------------------
def load_board(path: str) -> NpBoard:
    if path == "-" or path is None:
        lines = sys.stdin.read().splitlines()
    else:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    return parse_ascii(lines)

# ------------------------------------------------------------
# Wejście żarówek (kilka prostych formatów)
# - Format A (maskowy): R linii po W znaków: '.'=pusto, '*' lub 'b'='żarówka'
# - Format B (lista współrzędnych): linie "y x" (0-index), ewentualnie "x y" z flagą --xy
# - Format C (JSON): {"bulbs": [[y,x], ...]} lub {"bulbs_xy": [[x,y], ...]}
# ------------------------------------------------------------
def load_bulbs_mask(path: str, H: int, W: int, coords_are_xy: bool=False) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    if path is None:
        return mask  # domyślnie brak żarówek
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Spróbuj JSON
    if content.startswith("{"):
        try:
            obj = json.loads(content)
            if "bulbs" in obj:          # [[y,x], ...]
                for y, x in obj["bulbs"]:
                    mask[y, x] = True
                return mask
            if "bulbs_xy" in obj:       # [[x,y], ...]
                for x, y in obj["bulbs_xy"]:
                    mask[y, x] = True
                return mask
        except json.JSONDecodeError:
            pass

    # Spróbuj format maskowy (R linii ~ tej samej szerokości)
    lines = content.splitlines()
    if len(lines) == H and all(len(line) >= W for line in lines):
        ok_mask = True
        for y, line in enumerate(lines):
            for x, ch in enumerate(line[:W]):
                if ch in ('.', ' ', '0'):
                    continue
                elif ch in ('*', 'b', 'B', '1'):
                    mask[y, x] = True
                else:
                    # nieznany znak – uznaj, że to nie był format maskowy
                    ok_mask = False
                    break
            if not ok_mask:
                break
        if ok_mask:
            return mask

    # W przeciwnym razie traktuj jak listę współrzędnych
    # Domyślnie "y x", ale można przełączyć na "x y" flagą --xy
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        a, b = int(parts[0]), int(parts[1])
        if coords_are_xy:
            x, y = a, b
        else:
            y, x = a, b
        if 0 <= y < H and 0 <= x < W:
            mask[y, x] = True
    return mask

# ------------------------------------------------------------
# Losowe rozmieszczenie żarówek na pustych polach (do szybkich testów)
# ------------------------------------------------------------
def random_bulbs_mask(board: NpBoard, p: float, seed: int=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros((board.H, board.W), dtype=bool)
    # tylko na pustych polach
    draws = rng.random((board.H, board.W))
    mask[board.empty_mask] = draws[board.empty_mask] < p
    return mask

# ------------------------------------------------------------
# Główny program
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Evaluator funkcji celu dla Light Up/Akari (NumPy, wagi automatyczne)."
    )
    ap.add_argument("--board", "-b", required=True,
                    help="Ścieżka do pliku z planszą w ASCII (albo '-' dla STDIN).")
    ap.add_argument("--bulbs", "-u",
                    help="Opcjonalna ścieżka do żarówek (maskowy, lista współrzędnych albo JSON).")
    ap.add_argument("--xy", action="store_true",
                    help="Jeśli podano listę współrzędnych, interpretuje je jako 'x y' zamiast 'y x'.")
    ap.add_argument("--rand", type=float, default=None,
                    help="Jeśli podano, losowo rozmieść żarówki z prawdopodobieństwem p na pustych polach (0..1).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Ziarno RNG dla --rand.")
    ap.add_argument("--json", action="store_true",
                    help="Wypisz wynik w formacie JSON.")
    ap.add_argument("--show-weights", action="store_true",
                    help="Wypisz automatyczne wagi (w_conflict, w_digit, w_unlit).")
    # ... w argparse:
    ap.add_argument("--explain-digits", action="store_true",
                    help="Wypisz analizę każdej cyfry: pozycja, wartość, sąsiedzi, diff.")
    ap.add_argument("--neighbors", type=int, default=0,
                       help="Wypisz N sąsiadów bieżącego rozwiązania i zakończ (bez uruchamiania solvera).")
    ap.add_argument("--neighbors-strict", action="store_true",
                        help="Generuj sąsiadów z twardym zakazem konfliktów bulb-bulb.")
    ap.add_argument("--init-mode", choices=["soft", "strict", "digit_greedy"], default="soft",
                    help="Tryb losowej inicjalizacji: soft/strict/digit_greedy (gdy używasz --rand).")
    ap.add_argument("--fill", type=float, default=None,
                    help="Docelowy udział żarówek na pustych polach (zastępuje --rand).")

    args = ap.parse_args()

    # Plansza
    board = load_board(args.board)

    # Żarówki: priorytet -> --bulbs, potem --rand, w przeciwnym razie brak żarówek
    if args.bulbs:
        bulbs_mask = load_bulbs_mask(args.bulbs, board.H, board.W, coords_are_xy=args.xy)
    elif args.rand is not None:
        if not (0.0 <= args.rand <= 1.0):
            ap.error("--rand musi być w przedziale [0,1].")
        bulbs_mask = random_bulbs_mask(board, args.rand, seed=args.seed)
    else:
        bulbs_mask = np.zeros((board.H, board.W), dtype=bool)

    # Ocena
    try:
        c, d, u = penalties(board, bulbs_mask)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    wC, wD, wU = auto_weights(board)
    score = int(wC*c + wD*d + wU*u)

    # ... po wyliczeniu c,d,u i wag:
    if args.explain_digits:
        B = bulbs_mask
        up = np.zeros_like(B);
        up[1:, :] = B[:-1, :]
        down = np.zeros_like(B);
        down[:-1, :] = B[1:, :]
        left = np.zeros_like(B);
        left[:, 1:] = B[:, :-1]
        right = np.zeros_like(B);
        right[:, :-1] = B[:, 1:]
        neigh = up.astype(np.int16) + down.astype(np.int16) + left.astype(np.int16) + right.astype(np.int16)
        print("digits analysis:")
        for y in range(board.H):
            for x in range(board.W):
                if board.digit_mask[y, x]:
                    need = int(board.digit_val[y, x])
                    have = int(neigh[y, x])
                    print(f"  digit at (y={y}, x={x}) need={need} have={have} diff={abs(have - need)}")

    if args.json:
        out = {
            "score": score,
            "conflicts": int(c),
            "digit_errors": int(d),
            "unlit": int(u),
            "weights": {"conflict": int(wC), "digit": int(wD), "unlit": int(wU)},
            "H": board.H, "W": board.W,
            "bulbs_count": int(bulbs_mask.sum()),
        }
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(f"score:        {score}")
        print(f"conflicts:    {c}")
        print(f"digit_errors: {d}")
        print(f"unlit:        {u}")
        if args.show_weights:
            print(f"weights:      conflict={wC}, digit={wD}, unlit={wU}")
        print(f"bulbs_count:  {int(bulbs_mask.sum())}")
        print(f"size:         {board.H} x {board.W}")

    if args.neighbors > 0:
        rng = random.Random(getattr(args, "seed", None))
        base_score = int(score)  # score z bieżącej konfiguracji
        out = []
        bulbs = {(int(y), int(x)) for y, x in np.argwhere(bulbs_mask)}
        for cand in enumerate_neighbors(board, bulbs,
                                        max_neighbors=args.neighbors,
                                        rng=rng,
                                        use_strict=args.neighbors_strict,
                                        with_repairs=True):
            cand_mask = bulbs_set_to_mask(board, cand)
            c2, d2, u2 = penalties(board, cand_mask)
            s2 = int(auto_weights(board)[0] * c2 + auto_weights(board)[1] * d2 + auto_weights(board)[2] * u2)
            out.append({
                "bulbs": sorted(cand),
                "score": s2,
                "delta": int(s2 - base_score),
                "conflicts": int(c2),
                "digit_errors": int(d2),
                "unlit": int(u2),
            })
        # sortuj rosnąco po score (lepsze pierwsze)
        out.sort(key=lambda e: e["score"])
        print(json.dumps(out, ensure_ascii=False))
        return

if __name__ == "__main__":
    main()



# Przykłady użycia
#
# Czytanie planszy z pliku, brak żarówek
#
# python main.py --board plansza.txt
#
#
# Losowe żarówki (do szybkich testów)
#
# python main.py --board plansza.txt --rand 0.05 --seed 123 --show-weights
#
#
# Żarówki w masce (pliki o rozmiarze R×W, '*' oznacza żarówkę)
#
# python main.py --board plansza.txt --bulbs bulbs_mask.txt
#
#
# Żarówki jako lista współrzędnych „y x”
#
# python main.py --board plansza.txt --bulbs bulbs_coords.txt
#
#
# (dla listy „x y” dodaj --xy)
#
# Żarówki w JSON
#
# wariant [[y,x], ...]:
#
# {"bulbs": [[0,3], [4,1], [7,7]]}
#
#
# wariant [[x,y], ...]:
#
# {"bulbs_xy": [[3,0], [1,4], [7,7]]}
#
#
# Uruchom:
#
# python main.py --board plansza.txt --bulbs bulbs.json --json
