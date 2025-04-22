from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import List, Optional, Tuple
from fastapi.middleware.cors import CORSMiddleware
import math
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hằng số game
ROWS = 6
COLS = 7
WIN_LENGTH = 4
MAX_DEPTH = 6

class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]

class AIResponse(BaseModel):
    move: int
    evaluation: Optional[int] = None
    depth: Optional[int] = None
    execution_time: Optional[float] = None

class Connect4AI:
    # Biến lớp chứa tất cả các đường thắng
    WINNING_LINES = [
        # Horizontal
        [(r, c + i) for i in range(4)] for r in range(ROWS) for c in range(COLS - 3)
    ] + [
        # Vertical
        [(r + i, c) for i in range(4)] for c in range(COLS) for r in range(ROWS - 3)
    ] + [
        # Diagonal /
        [(r + i, c + i) for i in range(4)] for r in range(ROWS - 3) for c in range(COLS - 3)
    ] + [
        # Diagonal \
        [(r - i, c + i) for i in range(4)] for r in range(3, ROWS) for c in range(COLS - 3)
    ]

    @staticmethod
    def evaluate_window(window: List[int], player: int) -> int:
        """Đánh giá một cửa sổ 4 ô liên tiếp"""
        opponent = 3 - player
        score = 0
        
        if window.count(player) == 4:
            score += 100000
        elif window.count(player) == 3 and window.count(0) == 1:
            score += 500
        elif window.count(player) == 2 and window.count(0) == 2:
            score += 50
        elif window.count(player) == 2 and window.count(opponent) == 0 and window.count(0) == 2:
            score += 20
        
        if window.count(opponent) == 3 and window.count(0) == 1:
            score -= 400
        elif window.count(opponent) == 2 and window.count(0) == 2:
            score -= 20
        
        return score

    @staticmethod
    def detect_xxox_pattern(board: List[List[int]], player: int) -> List[Tuple[int, int]]:
        """Phát hiện mẫu XX_X hoặc X_XX cho người chơi"""
        threat_positions = []
        
        # Kiểm tra tất cả các hướng: ngang, dọc, và 2 đường chéo
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                # Kiểm tra XX_X
                if (c + 3 < COLS and 
                    board[r][c] == player and 
                    board[r][c+1] == player and 
                    board[r][c+2] == 0 and 
                    board[r][c+3] == player):
                    # Kiểm tra xem ô trống có đặt được không
                    empty_col = c + 2
                    empty_row = r
                    # Nếu không phải hàng cuối, kiểm tra xem có ô trống bên dưới không
                    if r < ROWS - 1:
                        if board[r+1][empty_col] != 0:  # Có quân cờ bên dưới
                            threat_positions.append((empty_row, empty_col))
                    else:  # Ở hàng cuối
                        threat_positions.append((empty_row, empty_col))
                
                # Kiểm tra X_XX
                if (c + 3 < COLS and 
                    board[r][c] == player and 
                    board[r][c+1] == 0 and 
                    board[r][c+2] == player and 
                    board[r][c+3] == player):
                    # Kiểm tra xem ô trống có đặt được không
                    empty_col = c + 1
                    empty_row = r
                    # Nếu không phải hàng cuối, kiểm tra xem có ô trống bên dưới không
                    if r < ROWS - 1:
                        if board[r+1][empty_col] != 0:  # Có quân cờ bên dưới
                            threat_positions.append((empty_row, empty_col))
                    else:  # Ở hàng cuối
                        threat_positions.append((empty_row, empty_col))
        
        # Vertical - chỉ có thể là X_XX từ dưới lên, XX_X không thể có từ dưới lên
        for c in range(COLS):
            for r in range(ROWS - 3):
                # Không cần kiểm tra XX_X vì quân cờ không thể lơ lửng
                
                # Kiểm tra X_XX (từ dưới lên)
                if (board[r][c] == player and 
                    board[r+1][c] == 0 and 
                    board[r+2][c] == player and 
                    board[r+3][c] == player):
                    # Vị trí này chắc chắn đặt được vì có quân phía dưới
                    threat_positions.append((r+1, c))
        
        # Diagonal /
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                # Kiểm tra XX_X
                if (board[r][c] == player and 
                    board[r+1][c+1] == player and 
                    board[r+2][c+2] == 0 and 
                    board[r+3][c+3] == player):
                    empty_row = r + 2
                    empty_col = c + 2
                    # Kiểm tra nếu có thể đặt tại vị trí trống
                    if empty_row == ROWS - 1 or board[empty_row+1][empty_col] != 0:
                        threat_positions.append((empty_row, empty_col))
                
                # Kiểm tra X_XX
                if (board[r][c] == player and 
                    board[r+1][c+1] == 0 and 
                    board[r+2][c+2] == player and 
                    board[r+3][c+3] == player):
                    empty_row = r + 1
                    empty_col = c + 1
                    # Kiểm tra nếu có thể đặt tại vị trí trống
                    if empty_row == ROWS - 1 or board[empty_row+1][empty_col] != 0:
                        threat_positions.append((empty_row, empty_col))
        
        # Diagonal \
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                # Kiểm tra XX_X
                if (board[r][c] == player and 
                    board[r-1][c+1] == player and 
                    board[r-2][c+2] == 0 and 
                    board[r-3][c+3] == player):
                    empty_row = r - 2
                    empty_col = c + 2
                    # Kiểm tra nếu có thể đặt tại vị trí trống
                    if empty_row == ROWS - 1 or board[empty_row+1][empty_col] != 0:
                        threat_positions.append((empty_row, empty_col))
                
                # Kiểm tra X_XX
                if (board[r][c] == player and 
                    board[r-1][c+1] == 0 and 
                    board[r-2][c+2] == player and 
                    board[r-3][c+3] == player):
                    empty_row = r - 1
                    empty_col = c + 1
                    # Kiểm tra nếu có thể đặt tại vị trí trống
                    if empty_row == ROWS - 1 or board[empty_row+1][empty_col] != 0:
                        threat_positions.append((empty_row, empty_col))
                        
        return threat_positions

    @staticmethod
    def count_open_threes(board: List[List[int]], player: int) -> int:
        """Đếm số lượng 'open threes' cho người chơi"""
        count = 0
        for line in Connect4AI.WINNING_LINES:
            pieces = [board[r][c] for r, c in line]
            if pieces.count(player) == 3 and pieces.count(0) == 1:
                for idx, piece in enumerate(pieces):
                    if piece == 0:
                        r, c = line[idx]
                        r_empty = Connect4AI.get_next_open_row(board, c)
                        if r_empty == r:
                            count += 1
                            break
        return count

    @staticmethod
    def evaluate_position(board: List[List[int]], player: int) -> int:
        """Đánh giá toàn bộ bảng cho người chơi hiện tại"""
        score = 0
        
        for c in range(COLS):
            for r in range(ROWS):
                if board[r][c] == player:
                    if c == 3:
                        score += 3
                    elif c in [2, 4]:
                        score += 2
        
        for r in range(ROWS):
            for c in range(COLS - 3):
                window = [board[r][c + i] for i in range(4)]
                score += Connect4AI.evaluate_window(window, player)
        
        for c in range(COLS):
            for r in range(ROWS - 3):
                window = [board[r + i][c] for i in range(4)]
                score += Connect4AI.evaluate_window(window, player)
        
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                window = [board[r + i][c + i] for i in range(4)]
                score += Connect4AI.evaluate_window(window, player)
        
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                window = [board[r - i][c + i] for i in range(4)]
                score += Connect4AI.evaluate_window(window, player)
        
        player_threes = Connect4AI.count_open_threes(board, player)
        opponent = 3 - player
        opponent_threes = Connect4AI.count_open_threes(board, opponent)
        
        score += player_threes * 1000
        if player_threes >= 2:
            score += 20000
        
        score -= opponent_threes * 1000
        if opponent_threes >= 2:
            score -= 20000
        
        # Đánh giá mẫu XX_X và X_XX
        player_xxox_patterns = Connect4AI.detect_xxox_pattern(board, player)
        opponent_xxox_patterns = Connect4AI.detect_xxox_pattern(board, opponent)
        
        score += len(player_xxox_patterns) * 5000
        score -= len(opponent_xxox_patterns) * 5000
        
        return score

    @staticmethod
    def is_terminal_node(board: List[List[int]]) -> bool:
        return Connect4AI.check_winner(board) != 0 or Connect4AI.is_board_full(board)

    @staticmethod
    def check_winner(board: List[List[int]]) -> int:
        for r in range(ROWS):
            for c in range(COLS - WIN_LENGTH + 1):
                if board[r][c] != 0 and all(board[r][c] == board[r][c+i] for i in range(1, WIN_LENGTH)):
                    return board[r][c]
        for r in range(ROWS - WIN_LENGTH + 1):
            for c in range(COLS):
                if board[r][c] != 0 and all(board[r][c] == board[r+i][c] for i in range(1, WIN_LENGTH)):
                    return board[r][c]
        for r in range(ROWS - WIN_LENGTH + 1):
            for c in range(COLS - WIN_LENGTH + 1):
                if board[r][c] != 0 and all(board[r][c] == board[r+i][c+i] for i in range(1, WIN_LENGTH)):
                    return board[r][c]
        for r in range(WIN_LENGTH - 1, ROWS):
            for c in range(COLS - WIN_LENGTH + 1):
                if board[r][c] != 0 and all(board[r][c] == board[r-i][c+i] for i in range(1, WIN_LENGTH)):
                    return board[r][c]
        return 0

    @staticmethod
    def is_board_full(board: List[List[int]]) -> bool:
        return all(board[0][c] != 0 for c in range(COLS))

    @staticmethod
    def get_valid_moves(board: List[List[int]]) -> List[int]:
        valid_moves = [c for c in range(COLS) if board[0][c] == 0]
        valid_moves.sort(key=lambda x: abs(x - COLS//2))
        return valid_moves

    @staticmethod
    def get_next_open_row(board: List[List[int]], col: int) -> int:
        for r in range(ROWS-1, -1, -1):
            if board[r][col] == 0:
                return r
        return -1

    @staticmethod
    def make_move(board: List[List[int]], col: int, player: int) -> Tuple[List[List[int]], int]:
        row = Connect4AI.get_next_open_row(board, col)
        if row == -1:
            return (board, -1)
        new_board = [row[:] for row in board]
        new_board[row][col] = player
        return (new_board, row)

    @staticmethod
    def minimax(board: List[List[int]], depth: int, alpha: float, beta: float, maximizing_player: bool, player: int) -> Tuple[int, Optional[int]]:
        valid_moves = Connect4AI.get_valid_moves(board)
        is_terminal = Connect4AI.is_terminal_node(board)
        
        if depth == 0 or is_terminal:
            if is_terminal:
                winner = Connect4AI.check_winner(board)
                if winner == player:
                    return (100000000000000, None)
                elif winner == 3 - player:
                    return (-100000000000000, None)
                else:
                    return (0, None)
            else:
                return (Connect4AI.evaluate_position(board, player), None)
        
        if maximizing_player:
            value = -math.inf
            best_move = valid_moves[0] if valid_moves else None
            for col in valid_moves:
                new_board, _ = Connect4AI.make_move(board, col, player)
                new_score, _ = Connect4AI.minimax(new_board, depth-1, alpha, beta, False, player)
                if new_score > value:
                    value = new_score
                    best_move = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return (value, best_move)
        else:
            value = math.inf
            best_move = valid_moves[0] if valid_moves else None
            for col in valid_moves:
                new_board, _ = Connect4AI.make_move(board, col, 3 - player)
                new_score, _ = Connect4AI.minimax(new_board, depth-1, alpha, beta, True, player)
                if new_score < value:
                    value = new_score
                    best_move = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return (value, best_move)

    @staticmethod
    def find_xxox_threats(board: List[List[int]], player: int) -> Optional[int]:
        """Tìm nước đi để chặn hoặc tạo mẫu XX_X/X_XX"""
        # Tìm mẫu có thể thắng của mình (XX_X hoặc X_XX)
        player_threats = Connect4AI.detect_xxox_pattern(board, player)
        if player_threats:
            for row, col in player_threats:
                # Kiểm tra xem có phải là nước đi tiếp theo hợp lệ không
                if Connect4AI.get_next_open_row(board, col) == row:
                    return col
        
        # Tìm mẫu cần chặn của đối thủ (XX_X hoặc X_XX)
        opponent = 3 - player
        opponent_threats = Connect4AI.detect_xxox_pattern(board, opponent)
        if opponent_threats:
            for row, col in opponent_threats:
                # Kiểm tra xem có phải là nước đi tiếp theo hợp lệ không
                if Connect4AI.get_next_open_row(board, col) == row:
                    return col
        
        return None

@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    try:
        if not game_state.valid_moves:
            raise ValueError("Không có nước đi hợp lệ")
        
        start_time = time.time()
        
        # Kiểm tra nước thắng ngay lập tức
        for col in game_state.valid_moves:
            new_board, row = Connect4AI.make_move(game_state.board, col, game_state.current_player)
            if row != -1 and Connect4AI.check_winner(new_board) == game_state.current_player:
                return AIResponse(
                    move=col,
                    evaluation=100,
                    depth=0,
                    execution_time=time.time() - start_time
                )
        
        # Kiểm tra nước chặn thắng của đối thủ
        opponent = 3 - game_state.current_player
        for col in game_state.valid_moves:
            new_board, row = Connect4AI.make_move(game_state.board, col, opponent)
            if row != -1 and Connect4AI.check_winner(new_board) == opponent:
                return AIResponse(
                    move=col,
                    evaluation=-100,
                    depth=0,
                    execution_time=time.time() - start_time
                )
        
        # Mới: Kiểm tra mẫu XX_X hoặc X_XX
        xxox_threat_move = Connect4AI.find_xxox_threats(game_state.board, game_state.current_player)
        if xxox_threat_move is not None and xxox_threat_move in game_state.valid_moves:
            return AIResponse(
                move=xxox_threat_move,
                evaluation=80,
                depth=0,
                execution_time=time.time() - start_time
            )
        
        # Chạy minimax nếu không có nước đi ưu tiên
        score, best_move = Connect4AI.minimax(
            game_state.board,
            MAX_DEPTH,
            -math.inf,
            math.inf,
            True,
            game_state.current_player
        )
        
        if best_move is None or best_move not in game_state.valid_moves:
            best_move = game_state.valid_moves[0]
        
        return AIResponse(
            move=best_move,
            evaluation=score,
            depth=MAX_DEPTH,
            execution_time=time.time() - start_time
        )
        
    except Exception as e:
        if game_state.valid_moves:
            return AIResponse(move=game_state.valid_moves[0])
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
@app.get("/api/test")
async def health_check():
    return {"status": "ok", "message": "Server is running"}