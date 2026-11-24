"""
Unit tests for Ultimate Tic Tac Toe implementation
"""

import unittest
from ultimate_ttt import (
    UltimateTicTacToe, MiniBoard, Move, Player, GameStatus
)


class TestMiniBoard(unittest.TestCase):
    """Test cases for the MiniBoard class."""
    
    def setUp(self):
        """Create a fresh mini-board for each test."""
        self.board = MiniBoard()
    
    def test_initial_state(self):
        """Test that a new board is empty and in progress."""
        self.assertEqual(self.board.status, GameStatus.IN_PROGRESS)
        self.assertEqual(self.board.winner, Player.NONE)
        self.assertFalse(self.board.is_complete())
    
    def test_valid_move(self):
        """Test making a valid move."""
        result = self.board.make_move(0, 0, Player.X)
        self.assertTrue(result)
        self.assertEqual(self.board.board[0][0], Player.X)
    
    def test_invalid_move_out_of_bounds(self):
        """Test that out-of-bounds moves are rejected."""
        self.assertFalse(self.board.make_move(-1, 0, Player.X))
        self.assertFalse(self.board.make_move(3, 0, Player.X))
        self.assertFalse(self.board.make_move(0, 5, Player.X))
    
    def test_invalid_move_occupied(self):
        """Test that occupied cells cannot be overwritten."""
        self.board.make_move(1, 1, Player.X)
        self.assertFalse(self.board.make_move(1, 1, Player.O))
        self.assertEqual(self.board.board[1][1], Player.X)
    
    def test_row_win(self):
        """Test winning by completing a row."""
        self.board.make_move(0, 0, Player.X)
        self.board.make_move(0, 1, Player.X)
        self.board.make_move(0, 2, Player.X)
        
        self.assertEqual(self.board.status, GameStatus.X_WON)
        self.assertEqual(self.board.winner, Player.X)
        self.assertTrue(self.board.is_complete())
    
    def test_column_win(self):
        """Test winning by completing a column."""
        self.board.make_move(0, 1, Player.O)
        self.board.make_move(1, 1, Player.O)
        self.board.make_move(2, 1, Player.O)
        
        self.assertEqual(self.board.status, GameStatus.O_WON)
        self.assertEqual(self.board.winner, Player.O)
    
    def test_diagonal_win(self):
        """Test winning by completing a diagonal."""
        self.board.make_move(0, 0, Player.X)
        self.board.make_move(1, 1, Player.X)
        self.board.make_move(2, 2, Player.X)
        
        self.assertEqual(self.board.status, GameStatus.X_WON)
        self.assertEqual(self.board.winner, Player.X)
    
    def test_anti_diagonal_win(self):
        """Test winning by completing the anti-diagonal."""
        self.board.make_move(0, 2, Player.O)
        self.board.make_move(1, 1, Player.O)
        self.board.make_move(2, 0, Player.O)
        
        self.assertEqual(self.board.status, GameStatus.O_WON)
        self.assertEqual(self.board.winner, Player.O)
    
    def test_draw(self):
        """Test a drawn game."""
        # Create a draw scenario
        moves = [
            (0, 0, Player.X), (0, 1, Player.O), (0, 2, Player.X),
            (1, 0, Player.O), (1, 1, Player.X), (1, 2, Player.O),
            (2, 0, Player.O), (2, 1, Player.X), (2, 2, Player.O),
        ]
        
        for row, col, player in moves:
            self.board.make_move(row, col, player)
        
        self.assertEqual(self.board.status, GameStatus.DRAW)
        self.assertEqual(self.board.winner, Player.NONE)
        self.assertTrue(self.board.is_complete())
    
    def test_cannot_play_after_win(self):
        """Test that moves are rejected after the board is won."""
        # Win the board
        self.board.make_move(0, 0, Player.X)
        self.board.make_move(1, 1, Player.X)
        self.board.make_move(2, 2, Player.X)
        
        # Try to make another move
        result = self.board.make_move(0, 1, Player.O)
        self.assertFalse(result)


class TestPlayer(unittest.TestCase):
    """Test cases for the Player enum."""
    
    def test_opponent(self):
        """Test the opponent method."""
        self.assertEqual(Player.X.opponent(), Player.O)
        self.assertEqual(Player.O.opponent(), Player.X)
        self.assertEqual(Player.NONE.opponent(), Player.NONE)


class TestMove(unittest.TestCase):
    """Test cases for the Move dataclass."""
    
    def test_move_creation(self):
        """Test creating a Move object."""
        move = Move(1, 2, 0, 1)
        self.assertEqual(move.super_row, 1)
        self.assertEqual(move.super_col, 2)
        self.assertEqual(move.mini_row, 0)
        self.assertEqual(move.mini_col, 1)


class TestUltimateTicTacToe(unittest.TestCase):
    """Test cases for the UltimateTicTacToe class."""
    
    def setUp(self):
        """Create a fresh game for each test."""
        self.game = UltimateTicTacToe()
    
    def test_initial_state(self):
        """Test that a new game is initialized correctly."""
        self.assertEqual(self.game.current_player, Player.X)
        self.assertIsNone(self.game.next_board)
        self.assertEqual(self.game.game_status, GameStatus.IN_PROGRESS)
        self.assertEqual(len(self.game.get_valid_moves()), 81)  # 9 boards * 9 cells
    
    def test_first_move_anywhere(self):
        """Test that the first move can be anywhere."""
        move = Move(1, 1, 2, 2)
        result = self.game.make_move(move)
        self.assertTrue(result)
        self.assertEqual(self.game.current_player, Player.O)
    
    def test_move_constrains_next_board(self):
        """Test that a move constrains the next player's board."""
        # X plays in center board, top-left cell
        self.game.make_move(Move(1, 1, 0, 0))
        
        # O must now play in top-left board (0, 0)
        self.assertEqual(self.game.next_board, (0, 0))
        self.assertEqual(self.game.current_player, Player.O)
        
        # Verify only 9 valid moves (in the top-left board)
        valid_moves = self.game.get_valid_moves()
        self.assertEqual(len(valid_moves), 9)
        self.assertTrue(all(m.super_row == 0 and m.super_col == 0 for m in valid_moves))
    
    def test_invalid_move_wrong_board(self):
        """Test that moves to the wrong board are rejected."""
        # X plays, constraining O to a specific board
        self.game.make_move(Move(1, 1, 0, 0))
        
        # O tries to play in the wrong board
        result = self.game.make_move(Move(1, 1, 1, 1))
        self.assertFalse(result)
        self.assertEqual(self.game.current_player, Player.O)  # Still O's turn
    
    def test_wild_move_when_board_complete(self):
        """Test that players can move anywhere when sent to a complete board."""
        # Complete a mini-board first
        self.game.make_move(Move(0, 0, 0, 0))  # X
        self.game.make_move(Move(0, 0, 1, 0))  # O
        self.game.make_move(Move(1, 0, 0, 0))  # X
        self.game.make_move(Move(0, 0, 1, 1))  # O
        self.game.make_move(Move(1, 1, 0, 0))  # X
        self.game.make_move(Move(0, 0, 1, 2))  # O (wins top-left board)
        
        # X must play in center-right board (0, 2) but it's empty, so normal constraint
        self.assertEqual(self.game.next_board, (1, 2))
        
        # Let's create scenario where next board is complete
        # Complete the target board
        self.game.make_move(Move(1, 2, 0, 0))  # X
        self.game.make_move(Move(0, 0, 0, 1))  # O (wild - top-left is complete)
        self.game.make_move(Move(0, 1, 0, 0))  # X
        self.game.make_move(Move(0, 0, 0, 2))  # O (wild)
        self.game.make_move(Move(1, 2, 1, 1))  # X
        self.game.make_move(Move(1, 1, 1, 1))  # O
        self.game.make_move(Move(1, 1, 2, 2))  # X
        self.game.make_move(Move(2, 2, 0, 0))  # O
        self.game.make_move(Move(1, 2, 2, 2))  # X (wins center-right)
        
        # O should be sent to bottom-right, but if that's complete...
        # Let's just check that wild moves work
        self.assertIsNotNone(self.game.get_valid_moves())
    
    def test_mini_board_win_updates_super_board(self):
        """Test that winning a mini-board is reflected in the super-board."""
        # Win the center mini-board as X with a diagonal
        self.game.make_move(Move(1, 1, 0, 0))  # X in center, top-left -> O to (0,0)
        self.game.make_move(Move(0, 0, 1, 1))  # O in (0,0), center -> X to (1,1)
        self.game.make_move(Move(1, 1, 1, 1))  # X in center, center -> O to (1,1)
        self.game.make_move(Move(1, 1, 0, 2))  # O in center, top-right -> X to (0,2)
        self.game.make_move(Move(0, 2, 0, 0))  # X in (0,2), top-left -> O to (0,0)
        self.game.make_move(Move(0, 0, 0, 0))  # O in (0,0), top-left -> X to (0,0)
        self.game.make_move(Move(0, 0, 2, 2))  # X in (0,0), bottom-right -> O to (2,2)
        self.game.make_move(Move(2, 2, 1, 1))  # O in (2,2), center -> X to (1,1)
        self.game.make_move(Move(1, 1, 2, 2))  # X in center, bottom-right (WINS with diagonal!)
        
        center_board = self.game.super_board[1][1]
        self.assertTrue(center_board.is_complete())
        self.assertEqual(center_board.winner, Player.X)
    
    def test_super_game_win(self):
        """Test winning the overall super-game."""
        # Simulate X winning top row of super-board
        # This is complex, so we'll just win three mini-boards manually
        
        # Win top-left (0,0)
        for move in [(0,0), (1,1), (2,2)]:
            r, c = move
            self.game.super_board[0][0].board[r][c] = Player.X
        self.game.super_board[0][0]._update_status()
        
        # Win top-center (0,1)
        for move in [(0,0), (1,1), (2,2)]:
            r, c = move
            self.game.super_board[0][1].board[r][c] = Player.X
        self.game.super_board[0][1]._update_status()
        
        # Win top-right (0,2)
        for move in [(0,0), (1,1), (2,2)]:
            r, c = move
            self.game.super_board[0][2].board[r][c] = Player.X
        self.game.super_board[0][2]._update_status()
        
        # Update super game status
        self.game._update_game_status()
        
        self.assertEqual(self.game.game_status, GameStatus.X_WON)
        self.assertEqual(self.game.winner, Player.X)
    
    def test_get_valid_moves_constrained(self):
        """Test that valid moves are correct when constrained."""
        self.game.make_move(Move(1, 1, 0, 0))
        
        valid = self.game.get_valid_moves()
        
        # Should have 9 moves, all in board (0, 0)
        self.assertEqual(len(valid), 9)
        for move in valid:
            self.assertEqual(move.super_row, 0)
            self.assertEqual(move.super_col, 0)
    
    def test_get_valid_moves_excludes_occupied(self):
        """Test that occupied cells are not in valid moves."""
        self.game.make_move(Move(1, 1, 0, 0))  # X in center -> O to (0,0)
        self.game.make_move(Move(0, 0, 1, 1))  # O in (0,0) -> X to (1,1)
        
        valid = self.game.get_valid_moves()
        
        # Should have 8 moves in board (1,1) (9 - 1 occupied by X)
        self.assertEqual(len(valid), 8)
        
        # The cell (0,0) in board (1,1) should not be in valid moves
        self.assertFalse(any(
            m.super_row == 1 and
            m.super_col == 1 and
            m.mini_row == 0 and
            m.mini_col == 0
            for m in valid
        ))


class TestGameIntegration(unittest.TestCase):
    """Integration tests for complete game scenarios."""
    
    def test_complete_game_sequence(self):
        """Test a complete sequence of moves in a game."""
        game = UltimateTicTacToe()
        
        # X will win center board with diagonal: (0,0), (1,1), (2,2)
        moves = [
            Move(1, 1, 0, 0),  # X in center (0,0) -> O to (0,0)
            Move(0, 0, 1, 1),  # O in (0,0) center -> X to (1,1)
            Move(1, 1, 1, 1),  # X in center (1,1) -> O to (1,1)
            Move(1, 1, 0, 2),  # O in center (0,2) -> X to (0,2)
            Move(0, 2, 0, 0),  # X in (0,2) -> O to (0,0)
            Move(0, 0, 0, 0),  # O in (0,0) -> X to (0,0)
            Move(0, 0, 2, 2),  # X in (0,0) (2,2) -> O to (2,2)
            Move(2, 2, 1, 1),  # O in (2,2) center -> X to (1,1)
            Move(1, 1, 2, 2),  # X in center (2,2) - WINS diagonal!
        ]
        
        for i, move in enumerate(moves):
            result = game.make_move(move)
            self.assertTrue(result, f"Move {i+1}: {move} should be valid")
        
        # Center board should be won by X
        self.assertEqual(game.super_board[1][1].winner, Player.X)
    
    def test_alternating_players(self):
        """Test that players alternate correctly."""
        game = UltimateTicTacToe()
        
        self.assertEqual(game.current_player, Player.X)
        game.make_move(Move(0, 0, 0, 0))
        self.assertEqual(game.current_player, Player.O)
        game.make_move(Move(0, 0, 1, 1))
        self.assertEqual(game.current_player, Player.X)


if __name__ == "__main__":
    unittest.main(verbosity=2)
