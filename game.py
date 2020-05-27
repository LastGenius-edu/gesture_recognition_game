#!/usr/bin/env python3
"""
Sultanov Andriy
MIT License 2020
"""
from tree import Tree
from board import TicTacToeBoard
import re
import rich


def computer_turn(board, player):
    """
    Decides on the best computer turn, makes a move and returns the board
    """
    markers = {"Player 1": "O", "Player 2": "X"}
    tree = Tree(board, player)
    tree.create_tree()
    tree.point_counter()
    position = tree.best_move()
    board[position] = markers[player]

    return board


def player_turn(board, player):
    """
    Asks the player for the cell to make a turn in, makes the move and returns the board
    """
    answer = ""
    markers = {"Player 1": "O", "Player 2": "X"}

    while answer == "":
        answer = input("Please enter the coordinates of your turn: ")
        if re.match(r"^\d+ *,? *\d+$", answer) is not None:
            col, row = answer.split()
            col, row = int(col), int(row)

            if board[(col, row)] != " ":
                print("This cell is not empty!")
                answer = ""
            else:
                board[(col, row)] = markers[player]
                return board
        else:
            answer = ""


def player_won():
    """
    Congratulates the player with the victory
    """
    rich.print(":tada:", ":tada:", ":confetti_ball:",
               "[bold green]Congratulations!!! You have won this game!!![/bold green]",
               ":confetti_ball:", ":tada:", ":tada:")


def draw():
    """
    Prints a draw message
    """
    rich.print("¯\_(ツ)_/¯",
               "[bold yellow]That's a draw... Maybe next time?[/bold yellow]",
               "¯\_(ツ)_/¯")


def computer_won():
    """
    Prints the failure message
    """
    rich.print(":cry:", ":cry:", ":worried:",
               "[bold red]You lost :( Maybe next time?[/bold red]",
               ":worried:", ":cry:", ":cry:")


def check_win(board, player):
    """
    Checks if anyone has won, if yes, stops the game and launches the congratulatory message.
    """
    if (result := board.calculate_result()) != 0 or board.free_cells() is None:
        if (result == 1 and player) or (result == -1 and not player):
            player_won()
        elif (result == 1 and not player) or (result == -1 and player):
            computer_won()
        else:
            draw()

        return True
    return False


def start_game():
    """
    Main game loop
    Starts the game and leads to the end
    """
    board = TicTacToeBoard()
    answer = ""
    rich.print("[bold blue]Welcome to Tic Tac Toe game![/bold blue]", ":game_die:")

    while len(answer) < 1 or answer[0].lower() not in ["y", "n"]:
        answer = input("Do you want to play first [Y/N]? ")

    print(board)

    if answer[0].lower() == "y":

        # Enter the game loop which will exit once the
        # board is filled or someone has won
        # The player's turn is first
        while True:
            player_turn(board, "Player 1")
            print(board)
            if check_win(board, True):
                break

            rich.print("[bold magenta]Computer's turn:[/bold magenta]")
            computer_turn(board, "Player 2")
            print(board)
            if check_win(board, True):
                break
    else:

        # Enter the game loop which will exit once the
        # board is filled or someone has won
        # The player's turn is second
        while True:
            print("Computer's turn:")
            computer_turn(board, "Player 1")
            print(board)
            if check_win(board, False):
                break

            player_turn(board, "Player 2")
            print(board)
            if check_win(board, False):
                break


if __name__ == '__main__':
    start_game()
