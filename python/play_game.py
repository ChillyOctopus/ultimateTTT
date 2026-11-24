"""
Interactive Ultimate Tic Tac Toe Game with Bot Support
Supports Player vs Player, Player vs Bot, and Bot vs Bot modes
"""

import os
import sys
import time
from ultimate_ttt import UltimateTicTacToe, Move, Player, GameStatus
from bot import Bot, OracleBot, RandomBot, CenterPreferringBot, CornerPreferringBot


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def get_player_move(game: UltimateTicTacToe, move_count: int) -> tuple:
    """
    Prompt the current player for their move.
    
    Args:
        game: The current game state
        move_count: The number of moves played so far
        
    Returns:
        Tuple of (Move object, command) where command is None or 'save'/'quit'
    """
    while True:
        try:
            print(f"{game.current_player.value}'s turn!")
            print(f"Move #{move_count + 1}")
            print("(Enter 'save' to save game, 'quit' to exit)")
            
            # Determine which boards can be played
            if game.next_board is not None:
                super_row, super_col = game.next_board
                super_num = Move.coords_to_num(super_row, super_col)
                user_input = input("> ").strip()
                
                # Check for commands
                if user_input.lower() == 'save':
                    return None, 'save'
                elif user_input.lower() == 'quit':
                    return None, 'quit'
                
                # Parse input - just one number
                if not user_input.isdigit():
                    print("❌ Please enter a single number 1-9")
                    continue
                
                mini_num = int(user_input)
                if not 1 <= mini_num <= 9:
                    print("❌ Please enter a number between 1-9")
                    continue
                
                move = Move.from_numbers(super_num, mini_num)
            else:
                user_input = input("> ").strip()
                
                # Check for commands
                if user_input.lower() == 'save':
                    return None, 'save'
                elif user_input.lower() == 'quit':
                    return None, 'quit'
                
                # Parse input
                parts = user_input.split()
                if len(parts) != 2:
                    print("❌ Please enter exactly 2 numbers")
                    continue
                
                if not all(p.isdigit() for p in parts):
                    print("❌ Please enter valid numbers")
                    continue
                
                super_num, mini_num = map(int, parts)
                
                if not (1 <= super_num <= 9 and 1 <= mini_num <= 9):
                    print("❌ Both numbers must be between 1-9")
                    continue
                
                move = Move.from_numbers(super_num, mini_num)
            
            # Check if it's in the valid moves list
            valid_moves = game.get_valid_moves()
            if move not in valid_moves:
                # Try to give helpful error message
                if game.next_board and (move.super_row, move.super_col) != game.next_board:
                    print(f"❌ You must play in mini-board {super_num}!")
                elif game.super_board[move.super_row][move.super_col].is_complete():
                    print("❌ That mini-board is already complete!")
                elif game.super_board[move.super_row][move.super_col].board[move.mini_row][move.mini_col] != Player.NONE:
                    print("❌ That cell is already occupied!")
                else:
                    print("❌ Invalid move!")
                continue
            
            return move, None
            
        except ValueError as e:
            print(f"❌ Error: {e}")
        except KeyboardInterrupt:
            print("\n\nGame interrupted!")
            sys.exit(0)


def select_game_mode():
    """
    Let the user select the game mode.
    
    Returns:
        Tuple of (x_bot, o_bot) where each is None for human or a Bot instance
    """
    clear_screen()
    print("=" * 50)
    print("  ULTIMATE TIC TAC TOE - GAME MODE SELECTION")
    print("=" * 50)
    print("\nSelect game mode:")
    print("  1. Player vs Player")
    print("  2. Player vs Bot")
    print("  3. Bot vs Bot")
    print("  4. Load saved game")
    print()
    
    while True:
        choice = input("Enter choice (1-4): ").strip()
        if choice == "1":
            return None, None, None
        elif choice == "2":
            bot_type = select_bot_type()
            # Ask which side the player wants
            print("\nDo you want to play as X or O?")
            while True:
                side = input("Enter X or O: ").strip().upper()
                if side == "X":
                    return None, create_bot_instance(bot_type, Player.O), None
                elif side == "O":
                    return create_bot_instance(bot_type, Player.X), None, None
                else:
                    print("Please enter X or O")
        elif choice == "3":
            print("\nSelect bot for X:")
            x_type = select_bot_type()
            print("\nSelect bot for O:")
            o_type = select_bot_type()
            return create_bot_instance(x_type, Player.X), create_bot_instance(o_type, Player.O), None
        elif choice == "4":
            return None, None, "load"
        else:
            print("Invalid choice. Please enter 1-4.")


def save_game(game: UltimateTicTacToe) -> bool:
    """
    Save the current game to a file.
    
    Args:
        game: The game to save
        
    Returns:
        True if saved successfully, False otherwise
    """
    print("\n" + "=" * 50)
    print("  SAVE GAME")
    print("=" * 50)
    
    default_name = f"uttt_save_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    if not default_name.endswith('.json'):
        default_name += '.json'
    
    if game.save_to_file(default_name):
        print(f"✅ Game saved to {default_name}")
        return True
    else:
        print("❌ Failed to save game")
        return False


def load_game() -> tuple:
    """
    Load a game from a file.
    
    Returns:
        Tuple of (game, x_bot, o_bot) or (None, None, None) if load failed
    """
    print("\n" + "=" * 50)
    print("  LOAD GAME")
    print("=" * 50)
    
    # List available save files
    import os
    import glob
    
    save_files = glob.glob("*.json")
    uttt_saves = [f for f in save_files if f.startswith("uttt_save_")]
    
    if uttt_saves:
        print("\nAvailable save files:")
        for i, f in enumerate(uttt_saves, 1):
            # Get file modification time
            mtime = os.path.getmtime(f)
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
            print(f"  {i}. {f} (saved: {timestamp})")
        print()
    
    filename = input("Enter filename to load: ").strip()
    
    if not filename:
        print("❌ No filename provided")
        return None, None, None
    
    game = UltimateTicTacToe.load_from_file(filename)
    
    if game is None:
        print("❌ Failed to load game")
        input("Press Enter to continue...")
        return None, None, None
    
    print(f"✅ Game loaded from {filename}")
    print("\nLoaded game state:")
    print(game)
    print()
    
    # Ask how to continue the game
    print("How do you want to continue this game?")
    print("  1. Player vs Player")
    print("  2. Player (X) vs Bot (O)")
    print("  3. Bot (X) vs Player (O)")
    print("  4. Bot vs Bot")
    print()
    
    while True:
        choice = input("Enter choice (1-4): ").strip()
        if choice == "1":
            return game, None, None
        elif choice == "2":
            bot_type = select_bot_type()
            return game, None, create_bot_instance(bot_type, Player.O)
        elif choice == "3":
            bot_type = select_bot_type()
            return game, create_bot_instance(bot_type, Player.X), None
        elif choice == "4":
            print("\nSelect bot for X:")
            x_type = select_bot_type()
            print("\nSelect bot for O:")
            o_type = select_bot_type()
            return game, create_bot_instance(x_type, Player.X), create_bot_instance(o_type, Player.O)
        else:
            print("Invalid choice. Please enter 1-4.")


def select_bot_type():
    """
    Let the user select the bot type.
    
    Returns:
        String representing the bot type
    """
    print("\nSelect bot type:")
    print("  1. Random Bot (completely random moves)")
    print("  2. Center Bot (prefers center positions)")
    print("  3. Corner Bot (prefers corner positions)")
    print("  4. Oracle Bot (perfect play - for testing only)")
    print()
    
    while True:
        choice = input("Enter choice (1-4): ").strip()
        if choice == "1":
            return "random"
        elif choice == "2":
            return "center"
        elif choice == "3":
            return "corner"
        elif choice == "4":
            return "oracle"
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


def create_bot_instance(bot_type: str, player: Player) -> Bot:
    """Create a bot instance of the specified type."""
    if bot_type == "random":
        return RandomBot(player)
    elif bot_type == "center":
        return CenterPreferringBot(player)
    elif bot_type == "corner":
        return CornerPreferringBot(player)
    elif bot_type == "oracle":
        return OracleBot(player)
    else:
        return RandomBot(player)


def play_game():
    """Main game loop for Ultimate Tic Tac Toe."""
    # Select game mode or load game
    result = select_game_mode()
    
    if len(result) == 3 and result[2] == "load":
        # Load a saved game
        game, x_bot, o_bot = load_game()
        if game is None:
            return  # Load failed, exit
    else:
        # Start a new game
        x_bot, o_bot = result[0], result[1]
        game = UltimateTicTacToe()
    
    clear_screen()
    print("=" * 50)
    print("  ULTIMATE TIC TAC TOE")
    print("=" * 50)
    
    # Show player configuration
    x_name = x_bot.name if x_bot else "Human"
    o_name = o_bot.name if o_bot else "Human"
    print(f"\nX: {x_name}")
    print(f"O: {o_name}")
    
    print("\nRules:")
    print("  - Win 3 mini-boards in a row to win the game")
    print("  - Where you play determines where your opponent plays next")
    print("  - The highlighted board (@) shows where you must play")
    print("  - If sent to a completed board, you can play anywhere")
    
    if not x_bot or not o_bot:  # At least one human player
        print("\nNumber pad system (like a phone/calculator):")
        print("  1  2  3")
        print("  4  5  6")
        print("  7  8  9")
        print("\n  - Use these numbers for both super-board and mini-board")
        print("  - Example: '5 1' = center super-board, top-left cell")
        print("  - Type 'save' to save game, 'quit' to exit")
    
    print("\nPress Enter to start...")
    input()
    
    move_count = 0
    bot_delay = 1.0  # Delay in seconds for bot moves (for visibility)
    
    while game.game_status == GameStatus.IN_PROGRESS:
        # time.sleep(3)
        # clear_screen()
        print(game)
        print()
        
        # Determine if current player is a bot
        current_bot = x_bot if game.current_player == Player.X else o_bot
        
        if current_bot:
            # Bot's turn
            print(f"{current_bot.name} ({game.current_player.value}) is thinking...")
            print(f"Move #{move_count + 1}")
            time.sleep(bot_delay)
            move = current_bot.get_move(game)
            time.sleep(3)
            command = None
            
            # Show the move
            super_num = Move.coords_to_num(move.super_row, move.super_col)
            mini_num = Move.coords_to_num(move.mini_row, move.mini_col)
            print(f"{current_bot.name} plays: Board {super_num}, Cell {mini_num}")
            
            # For bot vs bot, add a pause so user can see the move
            if x_bot and o_bot:
                time.sleep(bot_delay)
        else:
            # Human's turn
            move, command = get_player_move(game, move_count)
            
            # Handle commands
            if command == 'save':
                if save_game(game):
                    input("\nPress Enter to continue...")
                continue  # Don't increment move count, try again
            elif command == 'quit':
                print("\nQuitting game...")
                save_choice = input("Save before quitting? (y/n): ").strip().lower()
                if save_choice == 'y':
                    save_game(game)
                return
        
        if game.make_move(move):
            move_count += 1
        else:
            print("❌ Unexpected error: move validation failed!")
            input("Press Enter to continue...")
    
    # Game over - display final board
    clear_screen()
    print(game)
    print("\n" + "=" * 50)
    print(" " * 20 + "GAME OVER!" + " " * 20)
    print("=" * 50)
    
    if game.game_status == GameStatus.X_WON:
        winner_name = x_name if game.winner == Player.X else o_name
        print(f"\n🎉 {winner_name} (X) wins! 🎉")
    elif game.game_status == GameStatus.O_WON:
        winner_name = o_name if game.winner == Player.O else x_name
        print(f"\n🎉 {winner_name} (O) wins! 🎉")
    elif game.game_status == GameStatus.DRAW:
        print(f"\n🤝 It's a draw! 🤝")
    
    print(f"\nTotal moves: {move_count}")
    print("\nThanks for playing!")


if __name__ == "__main__":
    try:
        play_game()
    except KeyboardInterrupt:
        print("\n\nGame interrupted. Goodbye!")
        sys.exit(0)