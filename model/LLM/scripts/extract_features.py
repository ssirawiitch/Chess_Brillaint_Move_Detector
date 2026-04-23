import chess
import chess.pgn
import pandas as pd
import os
from tqdm import tqdm

def calculate_material(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    white_val = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in piece_values.items())
    black_val = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in piece_values.items())
    return white_val, black_val

def extract_eval_from_comment(comment):
    # Extracts [%eval x.xx] from comment
    if '[%eval ' in comment:
        try:
            start = comment.find('[%eval ') + 7
            end = comment.find(']', start)
            eval_str = comment[start:end].replace(',', '')
            # Handle mate e.g. #3 or #-2
            if '#' in eval_str:
                return float('inf') if eval_str.startswith('#') and '-' not in eval_str else float('-inf')
            return float(eval_str)
        except:
            return None
    return None

def process_pgn(pgn_path, output_path, max_games=None):
    print(f"Reading PGN from {pgn_path}...")
    
    data = []
    
    with open(pgn_path, 'r', encoding='utf-8') as pgn:
        games_processed = 0
        while True:
            if max_games and games_processed >= max_games:
                break
                
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            
            try:
                white_elo = int(game.headers.get("WhiteElo", 1500))
            except ValueError:
                white_elo = 1500
                
            try:
                black_elo = int(game.headers.get("BlackElo", 1500))
            except ValueError:
                black_elo = 1500
            
            board = game.board()
            prev_eval = None
            
            for node in game.mainline():
                move = node.move
                
                # 1. FEN ของกระดานก่อนเดิน
                fen = board.fen()
                
                # 2. ตาเดินนั้น
                san_move = board.san(move)
                
                # 3. is_check (whether the player is currently in check)
                is_check = int(board.is_check())
                
                # 4. material and elo
                white_mat, black_mat = calculate_material(board)
                if board.turn == chess.WHITE:
                    my_material = white_mat
                    opp_material = black_mat
                    player_elo = white_elo
                    opp_elo = black_elo
                else:
                    my_material = black_mat
                    opp_material = white_mat
                    player_elo = black_elo
                    opp_elo = white_elo
                
                num_legal_moves = board.legal_moves.count()
                
                # 5. delta_eval
                current_eval = extract_eval_from_comment(node.comment)
                delta_eval = None
                if prev_eval is not None and current_eval is not None and current_eval != float('inf') and current_eval != float('-inf') and prev_eval != float('inf') and prev_eval != float('-inf'):
                    # Change in evaluation after the move
                    # Evaluation is usually from White's perspective
                    delta_eval = current_eval - prev_eval
                    # If it was black's move, we want the delta to reflect black's gain
                    if board.turn == chess.BLACK:
                        delta_eval = -delta_eval
                
                # 6. is_brilliant
                # Check for NAG 3 (!!) or '!!' in comment
                is_brilliant = 1 if (3 in node.nags) or ('!!' in node.comment) or ('!!' in san_move) else 0
                
                data.append({
                    'fen': fen,
                    'move': san_move,
                    'player_elo': player_elo,
                    'elo_diff': player_elo - opp_elo,
                    'my_material': my_material,
                    'material_diff': my_material - opp_material,
                    'num_legal_moves': num_legal_moves,
                    'is_check': is_check,
                    'delta_eval': delta_eval,
                    'is_brilliant': is_brilliant
                })
                
                prev_eval = current_eval
                board.push(move)
            
            games_processed += 1
            if games_processed % 1000 == 0:
                print(f"Processed {games_processed} games...")

    print(f"Finished parsing. Total moves extracted: {len(data)}")
    df = pd.DataFrame(data)
    
    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == '__main__':
    pgn_file = '../data/lichess_db_standard_rated_2016-05_annotated.pgn'
    out_file = '../data/raw_features.csv'
    
    # Check if we should change working directory or use relative paths correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pgn_file = os.path.join(script_dir, '..', 'data', 'lichess_db_standard_rated_2016-05_annotated.pgn')
    out_file = os.path.join(script_dir, '..', 'data', 'raw_features.csv')
    
    if not os.path.exists(pgn_file):
        print(f"Error: PGN file not found at {pgn_file}")
    else:
        # Process all games in the PGN file
        process_pgn(pgn_file, out_file)
