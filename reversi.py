import numpy as np
import random
import time
import math
from copy import deepcopy

class ReversiBoard:
    """
    A class representing the Reversi game board and rules.
    """
    # Direction vectors for checking in all 8 directions
    DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    
    def __init__(self, size=8):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1  # 1 for black, 2 for white
        
        # Initialize the starting position
        mid = size // 2
        self.board[mid-1][mid-1] = 2  # White
        self.board[mid][mid] = 2      # White
        self.board[mid-1][mid] = 1    # Black
        self.board[mid][mid-1] = 1    # Black

    def get_opponent(self, player):
        """Return the opponent's number."""
        return 3 - player  # 1 -> 2, 2 -> 1
    
    def is_on_board(self, row, col):
        """Check if a position is on the board."""
        return 0 <= row < self.size and 0 <= col < self.size
    
    def count_pieces(self):
        """Count the number of pieces for each player."""
        black_count = np.sum(self.board == 1)
        white_count = np.sum(self.board == 2)
        return black_count, white_count
    
    def get_valid_moves(self, player=None):
        """Get all valid moves for the given player."""
        if player is None:
            player = self.current_player
        
        valid_moves = []
        
        for row in range(self.size):
            for col in range(self.size):
                if self.is_valid_move(row, col, player):
                    valid_moves.append((row, col))
        
        return valid_moves
    
    def is_valid_move(self, row, col, player=None):
        """Check if a move is valid for the given player."""
        if player is None:
            player = self.current_player
            
        # Cell must be empty
        if self.board[row][col] != 0:
            return False
        
        opponent = self.get_opponent(player)
        
        # Check in all eight directions
        for dx, dy in self.DIRECTIONS:
            r, c = row + dx, col + dy
            
            # There must be at least one opponent piece in this direction
            if not (self.is_on_board(r, c) and self.board[r][c] == opponent):
                continue
                
            # Continue in this direction
            r += dx
            c += dy
            
            # Keep going until we find an empty cell or go off the board
            while self.is_on_board(r, c) and self.board[r][c] != 0:
                # If we find our own piece, this is a valid move
                if self.board[r][c] == player:
                    return True
                    
                r += dx
                c += dy
        
        return False
    
    def make_move(self, row, col, player=None):
        """Make a move for the given player."""
        if player is None:
            player = self.current_player
            
        if not self.is_valid_move(row, col, player):
            return False
        
        # Place the piece
        self.board[row][col] = player
        
        # Flip opponent pieces
        opponent = self.get_opponent(player)
        pieces_to_flip = []
        
        for dx, dy in self.DIRECTIONS:
            r, c = row + dx, col + dy
            temp_flip = []
            
            # Check if there are opponent pieces to flip in this direction
            while self.is_on_board(r, c) and self.board[r][c] == opponent:
                temp_flip.append((r, c))
                r += dx
                c += dy
                
            # If we reached one of our own pieces, flip all pieces in between
            if self.is_on_board(r, c) and self.board[r][c] == player:
                pieces_to_flip.extend(temp_flip)
        
        # Flip all captured pieces
        for r, c in pieces_to_flip:
            self.board[r][c] = player
            
        # Switch players for next turn
        self.current_player = self.get_opponent(player)
        
        return True
    
    def has_valid_moves(self, player=None):
        """Check if the player has any valid moves."""
        if player is None:
            player = self.current_player
            
        return len(self.get_valid_moves(player)) > 0
    
    def is_game_over(self):
        """Check if the game is over."""
        return not (self.has_valid_moves(1) or self.has_valid_moves(2))
    
    def get_winner(self):
        """Get the winner of the game."""
        black_count, white_count = self.count_pieces()
        
        if black_count > white_count:
            return 1
        elif white_count > black_count:
            return 2
        else:
            return 0  # Draw
    
    def copy(self):
        """Create a deep copy of the board."""
        new_board = ReversiBoard(self.size)
        new_board.board = deepcopy(self.board)
        new_board.current_player = self.current_player
        return new_board
    
    def print_board(self, highlighted_pos=None):
        """
        Print the current board state with a nicer display.
        
        Args:
            highlighted_pos: Optional (row, col) tuple to highlight a position
        """
        # Clear the screen first (works on most terminals)
        print("\033c", end="")
        
        # Get valid moves for current player to mark with asterisk
        valid_moves = self.get_valid_moves()
        
        # Print column headers (A through H)
        col_headers = "    " + "   ".join([chr(65 + i) for i in range(self.size)])
        print(col_headers)
        
        # Print the top border
        print("  +" + "---+" * self.size)
        
        # Print each row
        for i in range(self.size):
            row_str = f"{i+1} |"
            for j in range(self.size):
                # Determine cell content
                if self.board[i][j] == 1:
                    cell = "1"  # Black
                elif self.board[i][j] == 2:
                    cell = "0"  # White
                elif (i, j) in valid_moves:
                    cell = "*"  # Valid move
                else:
                    cell = " "  # Empty
                
                # Highlight the cell if it's the selected position
                if highlighted_pos and highlighted_pos[0] == i and highlighted_pos[1] == j:
                    row_str += f"\033[7m {cell} \033[0m|"  # Inverse video for highlighting
                else:
                    row_str += f" {cell} |"
                    
            print(row_str)
            print("  +" + "---+" * self.size)
        
        black_count, white_count = self.count_pieces()
        print(f"1: Black: {black_count}  |  0: White: {white_count}")
        print(f"* marks valid moves for {self.current_player == 1 and 'Black' or 'White'}")


class MinimaxPlayer:
    """
    AI player using Minimax algorithm with Alpha-Beta pruning.
    """
    def __init__(self, player_number, depth=4):
        self.player_number = player_number
        self.depth = depth
        
        # Weights for the board evaluation
        self.weights = np.array([
            [120, -20, 20,  5,  5, 20, -20, 120],
            [-20, -40, -5, -5, -5, -5, -40, -20],
            [ 20,  -5, 15,  3,  3, 15,  -5,  20],
            [  5,  -5,  3,  3,  3,  3,  -5,   5],
            [  5,  -5,  3,  3,  3,  3,  -5,   5],
            [ 20,  -5, 15,  3,  3, 15,  -5,  20],
            [-20, -40, -5, -5, -5, -5, -40, -20],
            [120, -20, 20,  5,  5, 20, -20, 120]
        ])
    
    def get_move(self, board):
        """Get the best move using Minimax with Alpha-Beta pruning."""
        valid_moves = board.get_valid_moves(self.player_number)
        
        if not valid_moves:
            return None
        
        # If there's only one valid move, return it immediately
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        best_value = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        for move in valid_moves:
            # Make a copy of the board and make the move
            board_copy = board.copy()
            board_copy.make_move(move[0], move[1], self.player_number)
            
            # Get the value of this move
            value = self.minimax(board_copy, self.depth - 1, alpha, beta, False)
            
            if value > best_value:
                best_value = value
                best_move = move
                
            alpha = max(alpha, best_value)
        
        return best_move
    
    def minimax(self, board: ReversiBoard, depth, alpha, beta, is_maximizing):
        """
        Minimax algorithm with Alpha-Beta pruning.
        
        Args:
            board: Current board state
            depth: How many more levels to search
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            is_maximizing: Whether this is a maximizing level
            
        Returns:
            The score for the current board state
        """
        if depth == 0 or board.is_game_over():
            return self.evaluate(board)
        
        current_player = self.player_number if is_maximizing else 3 - self.player_number
        valid_moves = board.get_valid_moves(current_player)
        
        if not valid_moves:
            # If the current player has no valid moves, pass to the opponent
            board_copy = board.copy()
            board_copy.current_player = 3 - current_player
            return self.minimax(board_copy, depth - 1, alpha, beta, not is_maximizing)
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                board_copy = board.copy()
                board_copy.make_move(move[0], move[1], current_player)
                eval = self.minimax(board_copy, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                board_copy = board.copy()
                board_copy.make_move(move[0], move[1], current_player)
                eval = self.minimax(board_copy, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval
    
    def evaluate(self, board):
        """
        Evaluate the current board state from the perspective of this player.
        Higher scores are better for this player.
        """
        # If the game is over, return a high value if we won, low if we lost
        if board.is_game_over():
            winner = board.get_winner()
            if winner == self.player_number:
                return 10000  # We won
            elif winner == 0:
                return 0      # Draw
            else:
                return -10000  # We lost
        
        # Use a weighted matrix for position evaluation
        my_pieces = np.zeros((board.size, board.size), dtype=bool)
        opponent_pieces = np.zeros((board.size, board.size), dtype=bool)
        
        for i in range(board.size):
            for j in range(board.size):
                if board.board[i][j] == self.player_number:
                    my_pieces[i][j] = True
                elif board.board[i][j] == 3 - self.player_number:
                    opponent_pieces[i][j] = True
        
        # Calculate positional score
        my_score = np.sum(self.weights[my_pieces])
        opponent_score = np.sum(self.weights[opponent_pieces])
        
        # Calculate mobility score (number of valid moves)
        my_mobility = len(board.get_valid_moves(self.player_number))
        opponent_mobility = len(board.get_valid_moves(3 - self.player_number))
        
        # Combine scores with different weights
        positional_weight = 10
        mobility_weight = 5
        
        total_score = (positional_weight * (my_score - opponent_score) + 
                      mobility_weight * (my_mobility - opponent_mobility))
        
        return total_score


class MonteCarloNode:
    """A node in the Monte Carlo Search Tree."""
    def __init__(self, board: ReversiBoard, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move  # The move that led to this board state
        self.children: list[MonteCarloNode] = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = board.get_valid_moves()
    
    def select_child(self):
        """Select a child node using UCB1 formula."""
        # UCB1 formula: wins/visits + C * sqrt(ln(parent visits) / visits)
        C = 1.41  # Exploration parameter
        
        best_score = float('-inf')
        best_child = None
        
        for child in self.children:
            # Avoid division by zero
            if child.visits == 0:
                score = float('inf')
            else:
                exploitation = child.wins / child.visits
                exploration = C * math.sqrt(math.log(self.visits) / child.visits)
                score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def add_child(self, move):
        """Add a child node with the given move."""
        board_copy = self.board.copy()
        board_copy.make_move(move[0], move[1])
        
        child = MonteCarloNode(board_copy, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(child)
        
        return child
    
    def update(self, result):
        """Update this node with a simulation result."""
        self.visits += 1
        self.wins += result


class MCTSPlayer:
    """
    AI player using Monte Carlo Tree Search.
    """
    def __init__(self, player_number, iterations=1000):
        self.player_number = player_number
        self.iterations = iterations
    
    def get_move(self, board):
        """Get the best move using Monte Carlo Tree Search."""
        valid_moves = board.get_valid_moves(self.player_number)
        
        if not valid_moves:
            return None
        
        # If there's only one valid move, return it immediately
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Create the root node
        board_copy = board.copy()
        # Make sure it's our turn
        board_copy.current_player = self.player_number
        root = MonteCarloNode(board_copy)
        
        # Run MCTS iterations
        for _ in range(self.iterations):
            # Selection: Select a node to expand
            node = root
            while node.untried_moves == [] and node.children:
                node = node.select_child()
            
            # Expansion: Add a new child node
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node = node.add_child(move)
            
            # Simulation: Play a random game from this node
            board_copy = node.board.copy()
            while not board_copy.is_game_over():
                valid_moves = board_copy.get_valid_moves()
                if not valid_moves:
                    # If no valid moves, switch players
                    board_copy.current_player = 3 - board_copy.current_player
                    continue
                
                random_move = random.choice(valid_moves)
                board_copy.make_move(random_move[0], random_move[1])
            
            # Backpropagation: Update all nodes in the path
            winner = board_copy.get_winner()
            while node:
                if winner == self.player_number:
                    result = 1.0  # We won
                elif winner == 0:
                    result = 0.5  # Draw
                else:
                    result = 0.0  # We lost
                
                node.update(result)
                node = node.parent
        
        # Choose the move with the highest number of visits
        best_visits = -1
        best_move = None
        
        for child in root.children:
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = child.move
        
        return best_move

def play_interactive_game():
    """Play an interactive game against an AI opponent with interactive board navigation."""
    try:
        import msvcrt  # For Windows
        get_key_windows = True
    except ImportError:
        try:
            import tty, sys, termios  # For Unix
            get_key_windows = False
        except ImportError:
            print("Your platform doesn't support interactive mode. Using fallback input.")
            get_key_windows = None
            
    def get_key():
        """Get a single keypress from the user."""
        if get_key_windows is None:
            # Fallback - use regular input
            key = input("Enter your move (e.g. C4) or 'q' to quit: ")
            if key.lower() == 'q':
                return 'QUIT'
            return key
        elif get_key_windows:
            # Windows
            ch = msvcrt.getch()
            if ch == b'\xe0':  # Special key prefix
                ch = msvcrt.getch()
                if ch == b'H':  # Up arrow
                    return 'UP'
                elif ch == b'P':  # Down arrow
                    return 'DOWN'
                elif ch == b'K':  # Left arrow
                    return 'LEFT'
                elif ch == b'M':  # Right arrow
                    return 'RIGHT'
            elif ch == b'\r':  # Enter key
                return 'ENTER'
            elif ch == b' ':   # Space key
                return 'SPACE'
            elif ch == b'q' or ch == b'Q':  # Quit
                return 'QUIT'
            return ch.decode('utf-8')
        else:
            # Unix
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
                if ch == '\x1b':  # Escape sequence
                    ch = sys.stdin.read(2)
                    if ch == '[A':  # Up arrow
                        return 'UP'
                    elif ch == '[B':  # Down arrow
                        return 'DOWN'
                    elif ch == '[D':  # Left arrow
                        return 'LEFT'
                    elif ch == '[C':  # Right arrow
                        return 'RIGHT'
                elif ch == '\r':  # Enter key
                    return 'ENTER'
                elif ch == ' ':   # Space key
                    return 'SPACE'
                elif ch == 'q' or ch == 'Q':  # Quit
                    return 'QUIT'
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    print("Welcome to Reversi!")
    print("1. Play as Black against Minimax")
    print("2. Play as White against Minimax")
    print("3. Play as Black against MCTS")
    print("4. Play as White against MCTS")
    choice = int(input("Choose an option (1-4): "))
    
    board = ReversiBoard()
    
    if choice == 1:
        human_player = 1
        ai_player = MinimaxPlayer(2, depth=3)
        ai_name = "Minimax"
    elif choice == 2:
        human_player = 2
        ai_player = MinimaxPlayer(1, depth=3)
        ai_name = "Minimax"
    elif choice == 3:
        human_player = 1
        ai_player = MCTSPlayer(2, iterations=1000)
        ai_name = "MCTS"
    else:
        human_player = 2
        ai_player = MCTSPlayer(1, iterations=1000)
        ai_name = "MCTS"
    
    print(f"\nPlaying as {'Black' if human_player == 1 else 'White'} against {ai_name}.")
    print("Valid moves are marked with *")
    print("Use LEFT/RIGHT arrow keys to cycle through valid moves.")
    print("Press Enter or Space to confirm your move.")
    print("Press Q at any time to quit the game.")
    print("Press any key to start (or Q to quit)...")
    if get_key_windows is not None:
        key = get_key()
        if key == 'QUIT':
            print("\nExiting game...")
            return  # Exit the function to quit the game
    else:
        key = input()
        if key.lower() == 'q':
            print("\nExiting game...")
            return  # Exit the function to quit the game
    
    while not board.is_game_over():
        valid_moves = board.get_valid_moves()
        
        if board.current_player == human_player:
            if valid_moves:
                # Interactive move selection
                current_idx = 0
                highlight_pos = valid_moves[current_idx]
                
                board.print_board(highlight_pos)
                print(f"Your turn ({'Black' if human_player == 1 else 'White'})")
                print(f"Move {current_idx + 1}/{len(valid_moves)}: {chr(65 + highlight_pos[1])}{highlight_pos[0] + 1}")
                print("Use LEFT/RIGHT arrows to cycle through valid moves, ENTER to confirm")
                
                while True:
                    key = get_key()
                    
                    if key == 'QUIT':
                        print("\nExiting game...")
                        return  # Exit the function to quit the game
                    elif key == 'LEFT':
                        current_idx = (current_idx - 1) % len(valid_moves)
                        highlight_pos = valid_moves[current_idx]
                    elif key == 'RIGHT':
                        current_idx = (current_idx + 1) % len(valid_moves)
                        highlight_pos = valid_moves[current_idx]
                    elif key in ['ENTER', 'SPACE']:
                        # Confirm move
                        board.make_move(highlight_pos[0], highlight_pos[1], human_player)
                        break
                    
                    # Update the display
                    board.print_board(highlight_pos)
                    print(f"Your turn ({'Black' if human_player == 1 else 'White'})")
                    print(f"Move {current_idx + 1}/{len(valid_moves)}: {chr(65 + highlight_pos[1])}{highlight_pos[0] + 1}")
                    print("Use LEFT/RIGHT arrows to cycle through valid moves, ENTER to confirm")
            else:
                board.print_board()
                print("You have no valid moves. Passing...")
                print("Press any key to continue (or Q to quit)...")
                if get_key_windows is not None:
                    key = get_key()
                    if key == 'QUIT':
                        print("\nExiting game...")
                        return  # Exit the function to quit the game
                else:
                    key = input()
                    if key.lower() == 'q':
                        print("\nExiting game...")
                        return  # Exit the function to quit the game
                board.current_player = 3 - board.current_player
        else:
            board.print_board()
            if valid_moves:
                print(f"AI ({ai_name}) is thinking...")
                # Give the player a chance to see the board before AI moves
                time.sleep(0.5)
                move = ai_player.get_move(board)
                print(f"AI plays: {coord_to_algebraic(move[0], move[1])}")
                
                # Highlight the AI's move temporarily
                board.print_board(move)
                print(f"AI ({ai_name}) plays: {coord_to_algebraic(move[0], move[1])}")
                print("Press any key to continue (or Q to quit)...")
                
                # Wait for keypress before executing the move
                if get_key_windows is not None:
                    key = get_key()
                    if key == 'QUIT':
                        print("\nExiting game...")
                        return  # Exit the function to quit the game
                else:
                    key = input()
                    if key.lower() == 'q':
                        print("\nExiting game...")
                        return  # Exit the function to quit the game
                
                # Now actually make the move
                board.make_move(move[0], move[1])
                if get_key_windows is not None:
                    get_key()
                else:
                    input()
            else:
                print("AI has no valid moves. Passing...")
                print("Press any key to continue (or Q to quit)...")
                if get_key_windows is not None:
                    key = get_key()
                    if key == 'QUIT':
                        print("\nExiting game...")
                        return  # Exit the function to quit the game
                else:
                    key = input()
                    if key.lower() == 'q':
                        print("\nExiting game...")
                        return  # Exit the function to quit the game
                board.current_player = 3 - board.current_player
    
    # Game over
    board.print_board()
    winner = board.get_winner()
    black_count, white_count = board.count_pieces()
    
    print("Game over!")
    print(f"Final score - Black: {black_count}, White: {white_count}")
    
    if winner == 0:
        print("It's a draw!")
    elif winner == human_player:
        print("You win!")
    else:
        print("AI wins!")


def coord_to_algebraic(row, col):
    """Convert (row, col) coordinates to algebraic notation (e.g., 'C4')."""
    return f"{chr(65 + col)}{row + 1}"

def algebraic_to_coord(algebraic):
    """Convert algebraic notation (e.g., 'C4') to (row, col) coordinates."""
    col = ord(algebraic[0]) - 65
    row = int(algebraic[1:]) - 1
    return row, col

# Main program
if __name__ == "__main__":
    play_interactive_game()