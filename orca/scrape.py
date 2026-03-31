"""
Download games from hexo.did.science for SFT training.

Usage:
    python -m orca.scrape --games 1000 --output games.jsonl
    python -m orca.scrape --games 5000 --min-elo 1200 --output strong_games.jsonl
    python -m orca.scrape --games 100 --min-moves 20 --output long_games.jsonl
"""

import argparse
import json
import os
import sys
import time

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

BASE_URL = "https://hexo.did.science/api"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "HexBot-Trainer/3.0"})


def fetch_page(page: int, page_size: int = 20) -> list:
    """Fetch one page of finished games."""
    try:
        resp = SESSION.get(f"{BASE_URL}/finished-games", params={
            "page": page, "pageSize": page_size,
        }, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("games", data) if isinstance(data, dict) else data
    except Exception as e:
        print(f"  Error fetching page {page}: {e}")
        return []


def fetch_game_detail(game_id: str) -> dict:
    """Fetch full game details including moves."""
    try:
        resp = SESSION.get(f"{BASE_URL}/game/{game_id}", timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def scrape_games(
    output: str = "games.jsonl",
    max_games: int = 1000,
    min_elo: int = 0,
    min_moves: int = 6,
    delay: float = 0.5,
    verbose: bool = True,
) -> int:
    """Download games and save as JSONL.

    Returns number of games saved.
    """
    saved = 0
    page = 0
    seen_ids = set()

    # Load existing IDs to avoid duplicates
    if os.path.exists(output):
        with open(output, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    gid = record.get("id", record.get("gameId", ""))
                    if gid:
                        seen_ids.add(gid)
                except Exception:
                    pass
        if verbose:
            print(f"Found {len(seen_ids)} existing games in {output}")

    with open(output, 'a') as f:
        while saved < max_games:
            games = fetch_page(page)
            if not games:
                if verbose:
                    print(f"No more games at page {page}")
                break

            for game in games:
                if saved >= max_games:
                    break

                game_id = game.get("id", game.get("gameId", ""))
                if game_id in seen_ids:
                    continue

                # Filter by result
                result = game.get("gameResult", {})
                if result.get("reason") not in ("six-in-a-row", "resign", None):
                    continue

                # Fetch full game if needed
                detail = game
                moves = game.get("moves", [])
                if not moves and game_id:
                    detail = fetch_game_detail(game_id)
                    moves = detail.get("moves", [])
                    time.sleep(delay * 0.5)

                if len(moves) < min_moves:
                    continue

                # Filter by ELO
                players = detail.get("players", [])
                if min_elo > 0:
                    elos = [p.get("elo", 0) or 0 for p in players]
                    if elos and max(elos) < min_elo:
                        continue

                # Save
                f.write(json.dumps(detail) + "\n")
                f.flush()
                seen_ids.add(game_id)
                saved += 1

                if verbose and saved % 50 == 0:
                    print(f"  {saved}/{max_games} games saved")

            page += 1
            time.sleep(delay)

    if verbose:
        print(f"\nDone: {saved} games saved to {output}")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Download games from hexo.did.science for training")
    parser.add_argument("--output", "-o", default="games.jsonl",
                        help="Output file (default: games.jsonl)")
    parser.add_argument("--games", "-n", type=int, default=1000,
                        help="Number of games to download (default: 1000)")
    parser.add_argument("--min-elo", type=int, default=0,
                        help="Min player ELO (default: 0 = all)")
    parser.add_argument("--min-moves", type=int, default=6,
                        help="Min moves per game (default: 6)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between API calls in seconds (default: 0.5)")
    args = parser.parse_args()

    scrape_games(
        output=args.output,
        max_games=args.games,
        min_elo=args.min_elo,
        min_moves=args.min_moves,
        delay=args.delay,
    )


if __name__ == '__main__':
    main()
