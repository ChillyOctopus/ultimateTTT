"""
Ultimate Tic Tac Toe Game Implementation

This module implements the game logic for Ultimate Tic Tac Toe using
clean OOP design patterns and modular architecture.
"""

from enum import Enum
from typing import Optional, List, Tuple
from dataclasses import dataclass


class Player(Enum):
    """Represents the two players in the game."""
    X = "X"
    O = "O"
    NONE = " "
    
    def opponent(self) -> 'Player':
        """Returns the opponent of the current player."""
        if self == Player.X:
            return Player.O
        elif self == Player.O:
            return Player.X
        return Player.NONE


class GameStatus(Enum):
    """Represents the status of a game (mini or super)."""
    IN_PROGRESS = "in_progress"
    X_WON = "x_won"
    O_WON = "o_won"
    DRAW = "draw"


@dataclass
class Move:
    """Represents a move in the game."""
    super_row: int  # 0-2: which mini-game row in super board
    super_col: int  # 0-2: which mini-game col in super board
    mini_row: int   # 0-2: which cell row in mini-game
    mini_col: int   # 0-2: which cell col in mini-game
    
    def __str__(self) -> str:
        return f"Super({self.super_row},{self.super_col}) Mini({self.mini_row},{self.mini_col})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Move):
            return False
        return (self.super_row == other.super_row and
                self.super_col == other.super_col and
                self.mini_row == other.mini_row and
                self.mini_col == other.mini_col)
    
    def __hash__(self) -> int:
        return hash((self.super_row, self.super_col, self.mini_row, self.mini_col))
    
    @staticmethod
    def num_to_coords(num: int) -> Tuple[int, int]:
        """
        Convert number pad notation (1-9) to (row, col) coordinates.
        
        Number pad layout:
        1 2 3
        4 5 6
        7 8 9
        
        Maps to:
        (0,0) (0,1) (0,2)
        (1,0) (1,1) (1,2)
        (2,0) (2,1) (2,2)
        
        Args:
            num: Number from 1-9
            
        Returns:
            Tuple of (row, col)
        """
        if not 1 <= num <= 9:
            raise ValueError(f"Number must be 1-9, got {num}")
        num -= 1  # Convert to 0-8
        row = num // 3
        col = num % 3
        return (row, col)
    
    @staticmethod
    def coords_to_num(row: int, col: int) -> int:
        """
        Convert (row, col) coordinates to number pad notation (1-9).
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            
        Returns:
            Number from 1-9
        """
        if not (0 <= row < 3 and 0 <= col < 3):
            raise ValueError(f"Coordinates must be 0-2, got ({row}, {col})")
        return row * 3 + col + 1
    
    @classmethod
    def from_numbers(cls, super_num: int, mini_num: int) -> 'Move':
        """
        Create a Move from number pad notation.
        
        Args:
            super_num: Super board position (1-9)
            mini_num: Mini board position (1-9)
            
        Returns:
            Move object
        """
        super_row, super_col = cls.num_to_coords(super_num)
        mini_row, mini_col = cls.num_to_coords(mini_num)
        return cls(super_row, super_col, mini_row, mini_col)


class MiniBoard:
    """
    Represents a single 3x3 tic-tac-toe board (mini-game).
    Handles win detection and move validation for individual mini-games.
    """
    
    def __init__(self):
        """Initialize an empty 3x3 board."""
        self.board = [[Player.NONE for _ in range(3)] for _ in range(3)]
        self.status = GameStatus.IN_PROGRESS
        self.winner = Player.NONE
    
    def make_move(self, row: int, col: int, player: Player) -> bool:
        """
        Attempt to make a move on this mini-board.
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            player: The player making the move
            
        Returns:
            True if move was successful, False otherwise
        """
        if not self._is_valid_move(row, col):
            return False
        
        self.board[row][col] = player
        self._update_status()
        return True
    
    def _is_valid_move(self, row: int, col: int) -> bool:
        """Check if a move is valid (in bounds and cell is empty)."""
        if not (0 <= row < 3 and 0 <= col < 3):
            return False
        if self.status != GameStatus.IN_PROGRESS:
            return False
        return self.board[row][col] == Player.NONE
    
    def _update_status(self):
        """Check if the game has been won or drawn and update status."""
        winner = self._check_winner()
        if winner != Player.NONE:
            self.status = GameStatus.X_WON if winner == Player.X else GameStatus.O_WON
            self.winner = winner
        elif self._is_full():
            self.status = GameStatus.DRAW
    
    def _check_winner(self) -> Player:
        """
        Check if there's a winner on this board.
        
        Returns:
            The winning player or Player.NONE if no winner
        """
        # Check rows
        for row in range(3):
            if (self.board[row][0] == self.board[row][1] == self.board[row][2] 
                and self.board[row][0] != Player.NONE):
                return self.board[row][0]
        
        # Check columns
        for col in range(3):
            if (self.board[0][col] == self.board[1][col] == self.board[2][col] 
                and self.board[0][col] != Player.NONE):
                return self.board[0][col]
        
        # Check diagonals
        if (self.board[0][0] == self.board[1][1] == self.board[2][2] 
            and self.board[0][0] != Player.NONE):
            return self.board[0][0]
        
        if (self.board[0][2] == self.board[1][1] == self.board[2][0] 
            and self.board[0][2] != Player.NONE):
            return self.board[0][2]
        
        return Player.NONE
    
    def _is_full(self) -> bool:
        """Check if the board is completely filled."""
        return all(self.board[row][col] != Player.NONE 
                   for row in range(3) for col in range(3))
    
    def is_complete(self) -> bool:
        """Check if this mini-game is finished (won or drawn)."""
        return self.status != GameStatus.IN_PROGRESS
    
    def __str__(self) -> str:
        """Return a string representation of the mini-board."""
        lines = []
        for row in range(3):
            line = " | ".join(self.board[row][col].value for col in range(3))
            lines.append(line)
        return "\n".join(lines)


class UltimateTicTacToe:
    """
    Main game board for Ultimate Tic Tac Toe.
    Manages the super-board of 9 mini-games and enforces game rules.
    """
    
    def __init__(self):
        """Initialize the ultimate board with 9 mini-games."""
        self.super_board = [[MiniBoard() for _ in range(3)] for _ in range(3)]
        self.current_player = Player.X
        self.next_board: Optional[Tuple[int, int]] = None  # None means play anywhere
        self.game_status = GameStatus.IN_PROGRESS
        self.winner = Player.NONE
    
    def make_move(self, move: Move) -> bool:
        """
        Attempt to make a move on the ultimate board.
        
        Args:
            move: The Move object specifying where to play
            
        Returns:
            True if move was successful, False otherwise
        """
        if not self._is_valid_move(move):
            return False
        
        # Make the move on the mini-board
        mini_board = self.super_board[move.super_row][move.super_col]
        if not mini_board.make_move(move.mini_row, move.mini_col, self.current_player):
            return False
        
        # Update game state
        self._update_game_status()
        
        # Set the next board constraint
        next_super_row, next_super_col = move.mini_row, move.mini_col
        next_mini_board = self.super_board[next_super_row][next_super_col]
        
        if next_mini_board.is_complete():
            # Next player can play anywhere
            self.next_board = None
        else:
            # Next player must play in the specified mini-board
            self.next_board = (next_super_row, next_super_col)
        
        # Switch players
        self.current_player = self.current_player.opponent()
        
        return True
    
    def _is_valid_move(self, move: Move) -> bool:
        """
        Check if a move is valid according to Ultimate Tic Tac Toe rules.
        
        Args:
            move: The Move to validate
            
        Returns:
            True if the move is valid, False otherwise
        """
        if self.game_status != GameStatus.IN_PROGRESS:
            return False
        
        # Check if we're constrained to a specific mini-board
        if self.next_board is not None:
            required_row, required_col = self.next_board
            if (move.super_row != required_row or move.super_col != required_col):
                return False
        
        # Check if the mini-board is still playable
        mini_board = self.super_board[move.super_row][move.super_col]
        if mini_board.is_complete():
            return False
        
        return True
    
    def _update_game_status(self):
        """Check if the super-game has been won or drawn."""
        winner = self._check_super_winner()
        if winner != Player.NONE:
            self.game_status = GameStatus.X_WON if winner == Player.X else GameStatus.O_WON
            self.winner = winner
        elif self._is_super_draw():
            self.game_status = GameStatus.DRAW
    
    def _check_super_winner(self) -> Player:
        """
        Check if there's a winner on the super-board.
        
        Returns:
            The winning player or Player.NONE if no winner
        """
        # Check rows
        for row in range(3):
            winners = [self.super_board[row][col].winner for col in range(3)]
            if winners[0] == winners[1] == winners[2] and winners[0] != Player.NONE:
                return winners[0]
        
        # Check columns
        for col in range(3):
            winners = [self.super_board[row][col].winner for row in range(3)]
            if winners[0] == winners[1] == winners[2] and winners[0] != Player.NONE:
                return winners[0]
        
        # Check diagonals
        diag1 = [self.super_board[i][i].winner for i in range(3)]
        if diag1[0] == diag1[1] == diag1[2] and diag1[0] != Player.NONE:
            return diag1[0]
        
        diag2 = [self.super_board[i][2-i].winner for i in range(3)]
        if diag2[0] == diag2[1] == diag2[2] and diag2[0] != Player.NONE:
            return diag2[0]
        
        return Player.NONE
    
    def _is_super_draw(self) -> bool:
        """Check if the super-game is a draw (all mini-games complete, no winner)."""
        return all(self.super_board[row][col].is_complete() 
                   for row in range(3) for col in range(3))
    
    def get_valid_moves(self) -> List[Move]:
        """
        Get all valid moves for the current player.
        
        Returns:
            List of valid Move objects
        """
        valid_moves = []
        
        if self.game_status != GameStatus.IN_PROGRESS:
            return valid_moves
        
        # Determine which mini-boards we can play in
        if self.next_board is not None:
            # Constrained to a specific mini-board
            boards_to_check = [self.next_board]
        else:
            # Can play in any incomplete mini-board
            boards_to_check = [(row, col) for row in range(3) for col in range(3)]
        
        # Find all empty cells in valid mini-boards
        for super_row, super_col in boards_to_check:
            mini_board = self.super_board[super_row][super_col]
            if not mini_board.is_complete():
                for mini_row in range(3):
                    for mini_col in range(3):
                        if mini_board.board[mini_row][mini_col] == Player.NONE:
                            valid_moves.append(Move(super_row, super_col, mini_row, mini_col))
        
        return valid_moves
    
    def __str__(self) -> str:
        """Return a string representation of the entire ultimate board."""
        BORDER_CHAR = '.'
        HIGHLIGHT_CHAR = '@'
        lines = []
        # lines.append(f"Current Player: {self.current_player.value}")
        # if self.next_board:
        #     lines.append(f"Must play in mini-board: ({self.next_board[0]}, {self.next_board[1]})")
        # else:
        #     lines.append("Can play anywhere")
        # lines.append(f"Game Status: {self.game_status.value}")
        # lines.append("")
        
        for super_row in range(3):
            # Top border of this super-row
            border_line = "\t\t"
            for super_col in range(3):
                is_left_highlighted = (self.next_board == (super_row, super_col-1))
                is_top_highlighted = (self.next_board == (super_row-1, super_col))
                is_top_left_highlighted = (self.next_board == (super_row-1, super_col-1))
                is_highlighted = (self.next_board == (super_row, super_col))
                curr_border_char = HIGHLIGHT_CHAR if is_highlighted else BORDER_CHAR
                if is_highlighted or is_top_highlighted:
                    border_line += HIGHLIGHT_CHAR * 14
                elif is_left_highlighted or is_top_left_highlighted:
                    border_line += HIGHLIGHT_CHAR + BORDER_CHAR * 13
                else:
                    border_line += BORDER_CHAR * 14
            border_line += HIGHLIGHT_CHAR if is_highlighted or is_top_highlighted else BORDER_CHAR
            lines.append(border_line)
            
            # Spacer line (empty row at top of mini-boards)
            line = "\t\t"
            for super_col in range(3):
                is_left_highlighted = (self.next_board == (super_row, super_col-1))
                is_highlighted = (self.next_board == (super_row, super_col))
                side_char = HIGHLIGHT_CHAR if is_left_highlighted or is_highlighted else BORDER_CHAR
                line += f"{side_char}             "
            line += f"{HIGHLIGHT_CHAR if is_highlighted else BORDER_CHAR}"
            lines.append(line)
            
            # Each mini-board has 3 rows of cells
            for mini_row in range(3):

                # Cell content row
                line = "\t\t"
                for super_col in range(3):
                    mini_board = self.super_board[super_row][super_col]
                    is_left_highlighted = (self.next_board == (super_row, super_col-1))
                    is_highlighted = (self.next_board == (super_row, super_col))
                    side_char = HIGHLIGHT_CHAR if is_left_highlighted or is_highlighted else BORDER_CHAR
                    
                    # Show winner/draw status or the actual row
                    if mini_board.status == GameStatus.X_WON:
                        if mini_row == 0 or mini_row == 2:
                            cell_str = "  X       X  "
                        else:
                            cell_str = "      X      "
                    elif mini_board.status == GameStatus.O_WON:
                        if mini_row == 0 or mini_row == 2:
                            cell_str = "    O O O    "
                        else:
                            cell_str = "  O       O  "
                    elif mini_board.status == GameStatus.DRAW:
                        cell_str = "     DRAW    " if mini_row == 1 else "             "
                    else:
                        cells = [mini_board.board[mini_row][col].value for col in range(3)]
                        cell_str = f"  {cells[0]} | {cells[1]} | {cells[2]}  "
                    
                    line += f"{side_char}{cell_str}"
                line += f"{HIGHLIGHT_CHAR if is_highlighted else BORDER_CHAR}"
                lines.append(line)
                
                # Separator line within mini-board (if not last row)
                if mini_row < 2:
                    line = "\t\t"
                    for super_col in range(3):
                        mini_board = self.super_board[super_row][super_col]
                        is_left_highlighted = (self.next_board == (super_row, super_col-1))
                        is_highlighted = (self.next_board == (super_row, super_col))
                        side_char = HIGHLIGHT_CHAR if is_left_highlighted or is_highlighted else BORDER_CHAR
                        
                        # Show winner/draw status or the actual row
                        if mini_board.status == GameStatus.X_WON:
                            cell_str = "    X   X    "
                        elif mini_board.status == GameStatus.O_WON:
                            cell_str = "  O       O  "
                        else:
                            cell_str = " ----------- "

                        line += f"{side_char}{cell_str}"
                    line += f"{HIGHLIGHT_CHAR if is_highlighted else BORDER_CHAR}"
                    lines.append(line)
            
            # Spacer line (empty row at bottom of mini-boards)
            line = "\t\t"
            for super_col in range(3):
                is_left_highlighted = (self.next_board == (super_row, super_col-1))
                is_highlighted = (self.next_board == (super_row, super_col))
                side_char = HIGHLIGHT_CHAR if is_left_highlighted or is_highlighted else BORDER_CHAR
                line += f"{side_char}             "
            line += f"{HIGHLIGHT_CHAR if is_highlighted else BORDER_CHAR}"
            lines.append(line)

        # Bottom border of this super-row
        border_line = "\t\t"
        for super_col in range(3):
            is_left_highlighted = (self.next_board == (super_row, super_col-1))
            is_highlighted = (self.next_board == (super_row, super_col))
            curr_border_char = HIGHLIGHT_CHAR if is_highlighted else BORDER_CHAR
            if is_highlighted:
                border_line += HIGHLIGHT_CHAR * 14
            elif is_left_highlighted:
                border_line += HIGHLIGHT_CHAR + BORDER_CHAR * 13
            else:
                border_line += BORDER_CHAR * 14
        border_line += curr_border_char
        lines.append(border_line)
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """
        Convert the game state to a dictionary for saving.
        
        Returns:
            Dictionary representing the complete game state
        """
        state = {
            "current_player": self.current_player.value,
            "next_board": self.next_board,
            "game_status": self.game_status.value,
            "winner": self.winner.value,
            "super_board": []
        }
        
        # Save each mini-board
        for row in range(3):
            board_row = []
            for col in range(3):
                mini = self.super_board[row][col]
                mini_state = {
                    "board": [[cell.value for cell in mini_row] for mini_row in mini.board],
                    "status": mini.status.value,
                    "winner": mini.winner.value
                }
                board_row.append(mini_state)
            state["super_board"].append(board_row)
        
        return state
    
    @classmethod
    def from_dict(cls, state: dict) -> 'UltimateTicTacToe':
        """
        Create a game instance from a saved state dictionary.
        
        Args:
            state: Dictionary containing the game state
            
        Returns:
            UltimateTicTacToe instance with restored state
        """
        game = cls()
        
        # Restore top-level state
        game.current_player = Player(state["current_player"])
        game.next_board = tuple(state["next_board"]) if state["next_board"] else None
        game.game_status = GameStatus(state["game_status"])
        game.winner = Player(state["winner"])
        
        # Restore each mini-board
        for row in range(3):
            for col in range(3):
                mini_state = state["super_board"][row][col]
                mini = game.super_board[row][col]
                
                # Restore board cells
                for mini_row in range(3):
                    for mini_col in range(3):
                        mini.board[mini_row][mini_col] = Player(mini_state["board"][mini_row][mini_col])
                
                # Restore mini-board status
                mini.status = GameStatus(mini_state["status"])
                mini.winner = Player(mini_state["winner"])
        
        return game
    
    def save_to_file(self, filename: str) -> bool:
        """
        Save the current game state to a JSON file.
        
        Args:
            filename: Path to the save file
            
        Returns:
            True if save was successful, False otherwise
        """
        import json
        try:
            with open(filename, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving game: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, filename: str) -> Optional['UltimateTicTacToe']:
        """
        Load a game state from a JSON file.
        
        Args:
            filename: Path to the save file
            
        Returns:
            UltimateTicTacToe instance if successful, None otherwise
        """
        import json
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            return cls.from_dict(state)
        except FileNotFoundError:
            print(f"Save file not found: {filename}")
            return None
        except Exception as e:
            print(f"Error loading game: {e}")
            return None
    
    def copy(self) -> 'UltimateTicTacToe':
        """Create a deep copy of the game state (for simulation purposes)."""
        new_game = UltimateTicTacToe()
        new_game.current_player = self.current_player
        new_game.next_board = self.next_board
        new_game.game_status = self.game_status
        new_game.winner = self.winner
        
        for super_row in range(3):
            for super_col in range(3):
                mini_board = self.super_board[super_row][super_col]
                new_mini = MiniBoard()
                new_mini.board = [row[:] for row in mini_board.board]
                new_mini.status = mini_board.status
                new_mini.winner = mini_board.winner
                new_game.super_board[super_row][super_col] = new_mini
        
        return new_game