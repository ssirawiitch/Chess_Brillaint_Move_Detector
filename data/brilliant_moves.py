import sys
import os
import io
import time
import json
import concurrent.futures
import multiprocessing
from pathlib import Path

import chess
import chess.pgn
import chess.engine
import zstandard

os.environ['PYTHONIOENCODING'] = 'utf-8'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

"""
Brilliant Move Detector (Multiprocessing Version)
=================================================
Parses a PGN file and delegates batches of games to concurrent Stockfish processes.
"""

# ── Configuration ────────────────────────────────────────────────────────────
STOCKFISH_PATH = r"D:\Program Files (x86)\stockfish\stockfish-windows-x86-64-avx2.exe"
PGN_ZST_PATH  = r"C:\Users\USER\Downloads\brilliant-move-detector\data\lichess_db_standard_rated_2015-10.pgn"
OUTPUT_DIR     = r"C:\Users\USER\Downloads\brilliant-move-detector\output"

DEPTH          = 15        
MULTI_PV       = 2         
MAX_GAMES      = None      
SKIP_GAMES     = 14300      
FRESH_START    = True      
TIME_LIMIT     = 60.0      
BATCH_SIZE     = 10        

# ── Brilliant Move Thresholds ────────────────────────────────────────────────
MATERIAL_LOSS_THRESHOLD  = 0.5   
EVAL_TOLERANCE           = 0.50  
GAP_THRESHOLD            = 1.50  
# ─────────────────────────────────────────────────────────────────────────────

PIECE_VALUES = {
    chess.PAWN:   1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK:   5.0,
    chess.QUEEN:  9.0,
    chess.KING:   0.0,
}

def material_count(board: chess.Board, color: chess.Color) -> float:
    total = 0.0
    for piece_type in PIECE_VALUES:
        total += len(board.pieces(piece_type, color)) * PIECE_VALUES[piece_type]
    return total

def score_to_pawns(score: chess.engine.Score, board_turn: chess.Color) -> float:
    pov_score = score.white()
    if pov_score.is_mate():
        mate_in = pov_score.mate()
        return 100.0 if mate_in > 0 else -100.0
    cp = pov_score.score()
    if cp is None:
        return 0.0
    pawns = cp / 100.0
    if board_turn == chess.BLACK:
        pawns = -pawns
    return pawns

def clamp_score(s, cap=20.0):
    return max(-cap, min(cap, s))

def open_pgn(pgn_path: str):
    if pgn_path.endswith('.zst'):
        print(f"Opening {os.path.basename(pgn_path)} (streaming decompression)...")
        dctx = zstandard.ZstdDecompressor()
        fh = open(pgn_path, "rb")
        stream_reader = dctx.stream_reader(fh)
        text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8", errors="replace")
        return text_stream, fh
    else:
        print(f"Opening {os.path.basename(pgn_path)} (uncompressed)...")
        fh = open(pgn_path, "r", encoding="utf-8", errors="replace")
        return fh, fh  

def analyze_game(game: chess.pgn.Game, engine: chess.engine.SimpleEngine, depth: int, time_limit: float = None) -> list:
    brilliant_moves = []
    board = game.board()
    node = game

    move_num = 0
    for node in game.mainline():
        move = node.move
        move_num += 1
        color_name = "White" if board.turn == chess.WHITE else "Black"
        side = board.turn

        my_material_before = material_count(board, side)

        try:
            limit = chess.engine.Limit(depth=depth) if time_limit is None else chess.engine.Limit(depth=depth, time=time_limit)
            infos = engine.analyse(board, limit, multipv=MULTI_PV)
        except chess.engine.EngineTerminatedError:
            break

        if not infos or len(infos) == 0:
            board.push(move)
            continue

        best_info = infos[0]
        best_score = score_to_pawns(best_info["score"], side)

        second_score = None
        if len(infos) >= 2:
            second_score = score_to_pawns(infos[1]["score"], side)

        san = board.san(move)
        played_score = None
        for info in infos:
            if info.get("pv", [None])[0] == move:
                played_score = score_to_pawns(info["score"], side)
                break
        else:
            board.push(move)
            try:
                limit = chess.engine.Limit(depth=depth) if time_limit is None else chess.engine.Limit(depth=depth, time=time_limit)
                played_info = engine.analyse(board, limit)
                played_score = score_to_pawns(played_info["score"], side)
            except chess.engine.EngineTerminatedError:
                break
            board.pop()

        pv = best_info.get("pv", [])
        if not pv:
            pv = [move]

        sim_board = board.copy()
        sim_board.push(move)
        for pv_move in pv[1:5]: 
            if sim_board.is_legal(pv_move):
                sim_board.push(pv_move)

        my_material_future = material_count(sim_board, side)
        opp_material_before = material_count(board, not side)
        opp_material_future = material_count(sim_board, not side)

        my_loss = my_material_before - my_material_future
        opp_loss = opp_material_before - opp_material_future
        
        material_diff = my_loss - opp_loss

        white_eval = played_score if side == chess.WHITE else -played_score
        eval_str = f"[%eval {white_eval:+.2f}]"
        node.comment = f"{eval_str} {node.comment}".strip()

        is_sacrifice = material_diff >= MATERIAL_LOSS_THRESHOLD
        is_top_move = (best_score - played_score) <= EVAL_TOLERANCE
        has_gap = (second_score is not None) and (clamp_score(best_score) - clamp_score(second_score)) >= GAP_THRESHOLD

        if is_sacrifice and is_top_move and has_gap:
            full_move_num = board.fullmove_number
            brilliant = {
                "move_number": full_move_num,
                "color": color_name,
                "move_san": san,
                "material_sacrificed": round(material_diff, 2),
                "eval_best": round(best_score, 2),
                "eval_played": round(played_score, 2),
                "eval_second": round(second_score, 2) if second_score else None,
                "gap": round(best_score - second_score, 2) if second_score else None,
                "fen_before": board.fen(),
            }
            brilliant_moves.append(brilliant)
            node.nags.add(chess.pgn.NAG_BRILLIANT_MOVE)  

        board.push(move)

    return brilliant_moves

def process_game_batch(pgn_strings, depth, time_limit):
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 1, "Hash": 128}) 
    results = []
    
    try:
        for pgn_str in pgn_strings:
            game = chess.pgn.read_game(io.StringIO(pgn_str))
            
            brilliants = analyze_game(game, engine, depth, time_limit)
            
            white = game.headers.get("White", "?")
            black = game.headers.get("Black", "?")
            event = game.headers.get("Event", "?")
            date = game.headers.get("Date", "?")
            
            for b in brilliants:
                b["game"] = f"{white} vs {black}"
                b["event"] = event
                b["date"] = date

            pgn_str_annotated = str(game).replace(" $3", "!!")
            results.append((brilliants, pgn_str_annotated))
    finally:
        engine.quit()
        
    return results

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        pgn_io, raw_fh = open_pgn(PGN_ZST_PATH)
    except FileNotFoundError:
        print(f"Error: dataset file '{PGN_ZST_PATH}' not found.")
        return

    # ── Prepare Output Files ──
    if FRESH_START and SKIP_GAMES > 0:
        annotated_path = os.path.join(OUTPUT_DIR, f"annotated_games_from_{SKIP_GAMES}.pgn")
        report_path = os.path.join(OUTPUT_DIR, f"brilliant_moves_report_from_{SKIP_GAMES}.json")
    else:
        annotated_path = os.path.join(OUTPUT_DIR, "annotated_games.pgn")
        report_path = os.path.join(OUTPUT_DIR, "brilliant_moves_report.json")

    all_brilliant = []
    game_count = 0
    processed_games = 0

    if SKIP_GAMES == 0 or FRESH_START:
        with open(annotated_path, "w", encoding="utf-8") as f:
            pass

    if SKIP_GAMES > 0 and not FRESH_START and os.path.exists(report_path):
        print(f"\nLoading existing report to preserve previously found brilliant moves...")
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                old_report = json.load(f)
                all_brilliant = old_report.get("brilliant_moves", [])
                processed_games = old_report.get("summary", {}).get("games_analyzed", 0)
        except Exception:
            pass

    if SKIP_GAMES > 0:
        print(f"\nSkipping the first {SKIP_GAMES} games...")
        for _ in range(SKIP_GAMES):
            if not chess.pgn.skip_game(pgn_io):
                break
        print(f"Done skipping. Resuming from game {SKIP_GAMES + 1}...")

    max_workers = multiprocessing.cpu_count()
    
    print(f"\nStarting Multiprocessing Pipeline ({max_workers} Threads)...")
    start_main_time = time.time()
    
    def handle_completed_future(fut):
        nonlocal processed_games, all_brilliant, game_count
        try:
            batch_results = fut.result()
            
            for brilliants, pgn_str in batch_results:
                processed_games += 1
                all_brilliant.extend(brilliants)
                
                with open(annotated_path, "a", encoding="utf-8") as f:
                    print(pgn_str, file=f)
                    print(file=f)
                    
                for b in brilliants:
                    print(f"  !! {b['game']} - {b['move_number']}{'.' if b['color']=='White' else '...'}"
                          f"{b['move_san']}  (sacrifice: {b['material_sacrificed']:.1f}, gap: {b['gap']:.2f})")

            real_analyzed_count = processed_games + (SKIP_GAMES if not FRESH_START else 0)
            print(f"  [>] Processed {processed_games} games in this run... (Total found {len(all_brilliant)} brilliants)")
            
            report = {
                "config": {
                    "depth": DEPTH,
                    "material_loss_threshold": MATERIAL_LOSS_THRESHOLD,
                    "eval_tolerance": EVAL_TOLERANCE,
                    "gap_threshold": GAP_THRESHOLD,
                },
                "summary": {"games_analyzed": processed_games, "total_brilliant_moves": len(all_brilliant)},
                "brilliant_moves": all_brilliant,
            }
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

        except Exception as e:
            import traceback
            print(f"Error processing batch: {e}")
            traceback.print_exc()

    # The ProcessPoolExecutor is limited to keep memory usage healthy
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        active_futures = set()
        batch_strings = []

        while True:
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                break
            
            batch_strings.append(str(game))
            game_count += 1
            
            if len(batch_strings) >= BATCH_SIZE:
                fut = executor.submit(process_game_batch, batch_strings, DEPTH, TIME_LIMIT)
                active_futures.add(fut)
                batch_strings = []

            # Wait if there are enough futures to avoid running out of RAM
            while len(active_futures) >= max_workers * 2:
                done, active_futures = concurrent.futures.wait(
                    active_futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for f in done:
                    handle_completed_future(f)

            if MAX_GAMES is not None and game_count >= MAX_GAMES:
                break
                
        # Tail batch
        if batch_strings:
            fut = executor.submit(process_game_batch, batch_strings, DEPTH, TIME_LIMIT)
            active_futures.add(fut)
            
        raw_fh.close()
        
        # Flush remaining
        for fut in concurrent.futures.as_completed(active_futures):
            handle_completed_future(fut)

    elapsed_main = time.time() - start_main_time

    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(all_brilliant)} brilliant moves found across {processed_games} games")
    print(f"⏱TOTAL TIME: {elapsed_main:.1f} seconds")
    print(f"{'='*60}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
