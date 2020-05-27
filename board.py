#!/usr/bin/env python3
"""
Sultanov Andriy
MIT License 2020
"""


class TicTacToeBoard:
    """
    Board class for the TicTacToe game
    """
    possible_moves = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    def __init__(self, size=3, winning_length=3):
        """
        Initializes an empty board with size X size cells
        """
        assert isinstance(size, int), "Provide an integer value for the size"
        assert size > 0, "Size can't be zero"

        self.size = size
        self.winning_length = winning_length
        self.board = [[" " for _ in range(size)] for _ in range(size)]
        self.players = ["Player 1", "Player 2"]
        self.markers = {"Player 1": "O", "Player 2": "X"}

        # Represents the last turn: (symbol, position)
        self.last_turn = None

    def free_cells(self):
        """
        Returns a list of free cells
        None if none are free

        :return: list(tuple(int, int))
        """
        result = []
        for row_number in range(self.size):
            for col_number in range(self.size):
                if self[row_number, col_number] == " ":
                    result.append((row_number, col_number))

        if result:
            return result
        return None

    def __str__(self):
        """
        Returns a string representation of the game board

        :return: str
        """
        divider = "\n" + '-' * 3 * self.size + "\n"
        return f"{divider}{divider.join(' | '.join(row) for row in self.board)}{divider}"

    def __setitem__(self, cell_position, value):
        """
        Sets the cell at the board to the provided value of the player's marker

        :cell_position: (int, int)
        """
        if len(cell_position) != 2:
            raise IncorrectCellPositionError("Provide row and column numbers")
        for pos in cell_position:
            if not isinstance(pos, int):
                raise IncorrectCellPositionError("Provide integer row and column numbers")
            if not (0 <= pos < self.size):
                raise IncorrectCellPositionError("Provide valid row and column numbers")
        if self.board[cell_position[0]][cell_position[1]] != " ":
            raise CellNotEmptyError

        self.board[cell_position[0]][cell_position[1]] = value
        self.last_turn = (value, cell_position)

    def __getitem__(self, cell_position):
        """
        Returns the cell value at the given cell position

        :cell_position: (int, int)
        :return: str
        """
        if len(cell_position) != 2:
            raise IncorrectCellPositionError("Provide row and column numbers")
        for pos in cell_position:
            if not isinstance(pos, int):
                raise IncorrectCellPositionError("Provide integer row and column numbers")
            if not (0 <= pos < self.size):
                raise IncorrectCellPositionError("Provide valid row and column numbers")

        return self.board[cell_position[0]][cell_position[1]]

    def __find_winning_combination(self, cell_position):
        """
        Looks for a continuous line of the same kind of symbols of winning length
        starting from the given cell position
        Returns True if there is one, False otherwise

        :cell_position: (int, int)
        """
        start_value = self[cell_position]

        for direction in self.possible_moves:
            found_combination = True
            current_position = cell_position

            # Travel for winning_length cells in every possible side
            # If all the symbols are the same as the starting one, we've found a winning combination!
            for i in range(self.winning_length - 1):
                current_position = (current_position[0] + direction[0], current_position[1] + direction[1])
                try:
                    if self[current_position] != start_value:
                        found_combination = False
                        break
                except IncorrectCellPositionError:
                    found_combination = False
                    break

            if found_combination:
                return True

        # No winning combinations were found, return False
        return False

    def calculate_result(self):
        """
        Calculates the result of the game relative to the first player

        If the Player 1 won a game, returns 1
        If the Player 2 won a game, returns -1
        If there is a draw, returns 0
        """
        player_results = {player: False for player in self.players}

        for row in range(self.size):
            for col in range(self.size):
                if self[(row, col)] != " ":
                    if self.__find_winning_combination((row, col)):

                        # If the winning combination was found, change the player's status to winning
                        for player in self.players:
                            if self.markers[player] == self[(row, col)]:
                                player_results[player] = True

        player1 = player_results[self.players[0]]
        player2 = player_results[self.players[1]]
        if player1 and not player2:
            return 1
        if not player1 and player2:
            return -1
        return 0


class IncorrectCellPositionError(Exception):
    pass


class CellNotEmptyError(Exception):
    pass


if __name__ == '__main__':
    board = TicTacToeBoard()
    board[(0, 0)] = "X"
    board[(0, 1)] = "X"
    board[(0, 2)] = "X"
    print(board)
    print(board.calculate_result())
