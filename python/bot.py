"""
Bot players for Ultimate Tic Tac Toe
"""

import random
from typing import List
from ultimate_ttt import GameStatus, UltimateTicTacToe, Move, Player


class Bot:
    """Base class for Ultimate Tic Tac Toe bots."""
    
    def __init__(self, player: Player, name: str = "Bot"):
        """
        Initialize the bot.
        
        Args:
            player: Which player this bot controls (X or O)
            name: Display name for the bot
        """
        self.player = player
        self.name = name
    
    def get_move(self, game: UltimateTicTacToe) -> Move:
        """
        Get the bot's move for the current game state.
        
        Args:
            game: Current game state
            
        Returns:
            The move the bot wants to make
        """
        raise NotImplementedError("Subclasses must implement get_move")


class RandomBot(Bot):
    """Bot that makes completely random valid moves."""
    
    def __init__(self, player: Player, name: str = "Random Bot"):
        super().__init__(player, name)
    
    def get_move(self, game: UltimateTicTacToe) -> Move:
        """
        Select a random valid move.
        
        Args:
            game: Current game state
            
        Returns:
            A randomly selected valid move
        """
        valid_moves = game.get_valid_moves()
        
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        return random.choice(valid_moves)


class CenterPreferringBot(Bot):
    """Bot that prefers center positions but otherwise plays randomly."""
    
    def __init__(self, player: Player, name: str = "Center Bot"):
        super().__init__(player, name)
    
    def get_move(self, game: UltimateTicTacToe) -> Move:
        """
        Prefer center positions (position 5), otherwise random.
        
        Args:
            game: Current game state
            
        Returns:
            A move preferring center positions
        """
        valid_moves = game.get_valid_moves()
        
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        # Prefer center of mini-board (1,1)
        center_moves = [m for m in valid_moves if m.mini_row == 1 and m.mini_col == 1]
        if center_moves:
            return random.choice(center_moves)
        
        # Otherwise random
        return random.choice(valid_moves)


class CornerPreferringBot(Bot):
    """Bot that prefers corner positions but otherwise plays randomly."""
    
    def __init__(self, player: Player, name: str = "Corner Bot"):
        super().__init__(player, name)
    
    def get_move(self, game: UltimateTicTacToe) -> Move:
        """
        Prefer corner positions, otherwise random.
        
        Args:
            game: Current game state
            
        Returns:
            A move preferring corner positions
        """
        valid_moves = game.get_valid_moves()
        
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        # Prefer corners
        corner_positions = [(0, 0), (0, 2), (2, 0), (2, 2)]
        corner_moves = [m for m in valid_moves 
                       if (m.mini_row, m.mini_col) in corner_positions]
        if corner_moves:
            return random.choice(corner_moves)
        
        # Otherwise random
        return random.choice(valid_moves)
    
class OracleBot(Bot):
    """Bot that uses a perfect strategy (for testing purposes)."""
    
    def __init__(self, player: Player, name: str = "Oracle Bot"):
        self.counter = 0
        self.max_depth = 0
        super().__init__(player, name)

    def miniMax(self, game: UltimateTicTacToe, depth: int, maximizing_player: bool) -> int:
        """
        MiniMax algorithm to evaluate game states.
        
        Args:
            game: Current game state
            depth: How many moves ahead to evaluate
            maximizing_player: True if the current player is the maximizing player
            
        Returns:
            An integer score representing the desirability of the game state
        """
        
        valid_moves = game.get_valid_moves()
        self.counter += 1
        self.max_depth = max(self.max_depth, depth)
        print(f"Total evaluated: {self.counter}, Max depth: {self.max_depth}")
        
        # Base case: check for terminal state or depth limit
        if game.game_status != GameStatus.IN_PROGRESS or depth == -1:
            if game.game_status == GameStatus.X_WON:
                return 1 if self.player == Player.X else -1
            elif game.game_status == GameStatus.O_WON:
                return 1 if self.player == Player.O else -1
            else:
                return 0  # Draw or non-terminal
        if maximizing_player:
            max_eval = float('-inf')
            for move in valid_moves:
                new_game = game.copy()
                new_game.make_move(move)
                eval = self.miniMax(new_game, depth + 1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                new_game = game.copy()
                new_game.make_move(move)
                eval = self.miniMax(new_game, depth + 1, True)
                min_eval = min(min_eval, eval)
            return min_eval
    
    def get_move(self, game: UltimateTicTacToe) -> Move:
        """
        Use a perfect strategy to select the best move.
        
        Args:
            game: Current game state
            
        Returns:
            The best move according to a perfect strategy
        """
        valid_moves = game.get_valid_moves()
        
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        best_move = None
        best_score = float('-inf') if self.player == Player.X else float('inf')
        
        for move in valid_moves:
            new_game = game.copy()
            new_game.make_move(move)
            score = self.miniMax(new_game, depth=0, maximizing_player=(self.player == Player.X))
            print(f"Evaluating move: {move} -> Score: {score}")
            if self.player == Player.X and score > best_score:
                best_score = score
                best_move = move
            elif self.player == Player.O and score < best_score:
                best_score = score
                best_move = move
        
        return best_move


def create_bot(bot_type: str, player: Player) -> Bot:
    """
    Factory function to create bots.
    
    Args:
        bot_type: Type of bot ("random", "center", "corner")
        player: Which player the bot controls
        
    Returns:
        A Bot instance
    """
    bot_types = {
        "random": RandomBot,
        "center": CenterPreferringBot,
        "corner": CornerPreferringBot,
        "oracle": OracleBot
    }
    
    if bot_type.lower() not in bot_types:
        raise ValueError(f"Unknown bot type: {bot_type}. Choose from: {list(bot_types.keys())}")
    
    return bot_types[bot_type.lower()](player)