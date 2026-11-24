"""
Bot Simulation Script
Watch bots play Ultimate Tic Tac Toe with configurable settings
"""

import time
import sys
from ultimate_ttt import UltimateTicTacToe, Move, Player, GameStatus
from bot import OracleBot, RandomBot, CenterPreferringBot, CornerPreferringBot
import os


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')

def simulate_game(x_bot, o_bot, delay=0.5, show_moves=True):
    """
    Simulate a complete game between two bots.
    
    Args:
        x_bot: Bot playing as X
        o_bot: Bot playing as O
        delay: Seconds to pause between moves (0 for no delay)
        show_moves: Whether to print the board after each move
        
    Returns:
        Tuple of (winner, move_count) where winner is Player.X, Player.O, or Player.NONE for draw
    """
    game = UltimateTicTacToe()
    move_count = 0
    
    print(f"\n{'='*50}")
    print(f"  {x_bot.name} vs {o_bot.name}")
    print(f"{'='*50}\n")
    
    if show_moves:
        clear_screen()
        print(game)
        if delay > 0:
            time.sleep(delay)
    
    while game.game_status == GameStatus.IN_PROGRESS:
        current_bot = x_bot if game.current_player == Player.X else o_bot
        move = current_bot.get_move(game)
        
        if show_moves:
            super_num = Move.coords_to_num(move.super_row, move.super_col)
            mini_num = Move.coords_to_num(move.mini_row, move.mini_col)
            print(f"\nMove {move_count + 1}: {current_bot.name} plays Board {super_num}, Cell {mini_num}")
        
        game.make_move(move)
        move_count += 1
        
        if show_moves:
            clear_screen()
            print(game)
            if delay > 0:
                time.sleep(delay)
    
    # Game over
    print(f"\n{'='*50}")
    print(f"  GAME OVER - {move_count} moves")
    print(f"{'='*50}")
    
    if game.game_status == GameStatus.X_WON:
        print(f"🎉 {x_bot.name} wins!")
        return Player.X, move_count
    elif game.game_status == GameStatus.O_WON:
        print(f"🎉 {o_bot.name} wins!")
        return Player.O, move_count
    else:
        print(f"🤝 Draw!")
        return Player.NONE, move_count


def run_tournament(bot_types, games_per_matchup=10):
    """
    Run a round-robin tournament between different bot types.
    
    Args:
        bot_types: List of tuples (bot_class, name)
        games_per_matchup: Number of games each pair plays
    """
    print(f"\n{'='*60}")
    print(f"  ULTIMATE TIC TAC TOE BOT TOURNAMENT")
    print(f"  {games_per_matchup} games per matchup")
    print(f"{'='*60}\n")
    
    results = {}
    for bot_class, name in bot_types:
        results[name] = {"wins": 0, "losses": 0, "draws": 0, "total_moves": 0}
    
    matchup_count = 0
    total_matchups = len(bot_types) * (len(bot_types) - 1) // 2
    
    for i, (bot1_class, bot1_name) in enumerate(bot_types):
        for bot2_class, bot2_name in bot_types[i+1:]:
            matchup_count += 1
            print(f"\nMatchup {matchup_count}/{total_matchups}: {bot1_name} vs {bot2_name}")
            print("-" * 50)
            
            matchup_results = {bot1_name: 0, bot2_name: 0, "draws": 0}
            
            for game_num in range(games_per_matchup):
                # Alternate who plays X
                if game_num % 2 == 0:
                    x_bot = bot1_class(Player.X, bot1_name)
                    o_bot = bot2_class(Player.O, bot2_name)
                    x_name, o_name = bot1_name, bot2_name
                else:
                    x_bot = bot2_class(Player.X, bot2_name)
                    o_bot = bot1_class(Player.O, bot1_name)
                    x_name, o_name = bot2_name, bot1_name
                
                # Play the game silently
                winner, move_count = simulate_game(x_bot, o_bot, delay=0, show_moves=False)
                
                # Record results
                if winner == Player.X:
                    matchup_results[x_name] += 1
                    results[x_name]["wins"] += 1
                    results[o_name]["losses"] += 1
                    results[x_name]["total_moves"] += move_count
                    results[o_name]["total_moves"] += move_count
                elif winner == Player.O:
                    matchup_results[o_name] += 1
                    results[o_name]["wins"] += 1
                    results[x_name]["losses"] += 1
                    results[x_name]["total_moves"] += move_count
                    results[o_name]["total_moves"] += move_count
                else:
                    matchup_results["draws"] += 1
                    results[bot1_name]["draws"] += 1
                    results[bot2_name]["draws"] += 1
                    results[bot1_name]["total_moves"] += move_count
                    results[bot2_name]["total_moves"] += move_count
            
            print(f"{bot1_name}: {matchup_results[bot1_name]} wins")
            print(f"{bot2_name}: {matchup_results[bot2_name]} wins")
            print(f"Draws: {matchup_results['draws']}")
    
    # Print final standings
    print(f"\n{'='*60}")
    print(f"  FINAL STANDINGS")
    print(f"{'='*60}\n")
    
    print(f"{'Bot':<20} {'Wins':<8} {'Losses':<8} {'Draws':<8} {'Avg Moves':<10}")
    print("-" * 60)
    
    sorted_bots = sorted(bot_types, key=lambda x: results[x[1]]["wins"], reverse=True)
    for bot_class, name in sorted_bots:
        stats = results[name]
        total_games = stats["wins"] + stats["losses"] + stats["draws"]
        avg_moves = stats["total_moves"] / total_games if total_games > 0 else 0
        print(f"{name:<20} {stats['wins']:<8} {stats['losses']:<8} {stats['draws']:<8} {avg_moves:<10.1f}")


if __name__ == "__main__":
    # Define available bots
    bots = [
        (RandomBot, "Random Bot"),
        (CenterPreferringBot, "Center Bot"),
        (CornerPreferringBot, "Corner Bot"),
    ]
    
    print("Ultimate Tic Tac Toe Bot Simulator")
    print("\nOptions:")
    print("  1. Watch a single game (with delay)")
    print("  2. Run quick tournament (10 games per matchup)")
    print("  3. Run full tournament (1000 games per matchup)")
    print()
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        print("\nSelect Bot 1:")
        print("  1. Random Bot")
        print("  2. Center Bot")
        print("  3. Corner Bot")
        print("  4. Oracle Bot")
        bot1_choice = input("Enter choice (1-4): ").strip()
        
        print("\nSelect Bot 2:")
        print("  1. Random Bot")
        print("  2. Center Bot")
        print("  3. Corner Bot")
        print("  4. Oracle Bot")
        bot2_choice = input("Enter choice (1-4): ").strip()
        
        bot_map = {"1": RandomBot, "2": CenterPreferringBot, "3": CornerPreferringBot, "4": OracleBot}
        name_map = {"1": "Random Bot", "2": "Center Bot", "3": "Corner Bot", "4": "Oracle Bot"}
        
        if bot1_choice in bot_map and bot2_choice in bot_map:
            x_bot = bot_map[bot1_choice](Player.X, name_map[bot1_choice] + " X")
            o_bot = bot_map[bot2_choice](Player.O, name_map[bot2_choice] + " O")
            simulate_game(x_bot, o_bot, delay=0, show_moves=True)
        else:
            print("Invalid selection!")
    
    elif choice == "2":
        run_tournament(bots, games_per_matchup=10)
    
    elif choice == "3":
        print("\nRunning full tournament... This may take a minute.")
        run_tournament(bots, games_per_matchup=1000)
    
    else:
        print("Invalid choice!")