#!/usr/bin/env python3
"""
Sultanov Andriy
MIT License 2020
"""
from node import Node
from copy import deepcopy


class Tree:
    """
    Tree for the TicTacToe game, representing a set
    of possible turns and consequent game boards
    """

    def __init__(self, starting_board, player="Player 1"):
        self._root = Node(starting_board)
        self.player = player
        self.markers = {"Player 1": "O", "Player 2": "X"}

    def create_tree(self):
        """
        Builds a poly tree of possible boards.
        Looks at every available turn option
        """
        def recurse(node, player):
            current_board = node.board
            possible_turns = current_board.free_cells()

            if possible_turns is None:
                return None

            else:
                for turn in possible_turns:
                    new_board = deepcopy(current_board)

                    new_board[turn] = self.markers[player]

                    node.children.append(Node(new_board))

                for child in node.children:
                    recurse(child, "Player 1" if player == "Player 2" else "Player 2")

        recurse(self._root, self.player)

    def point_counter(self):
        """
        Recursively counts points for the subtree.
        """
        def recurse(node):
            for child in node.children:
                node.points += recurse(child)

            if not node.children:
                match_result = node.board.calculate_result()
                if match_result == 1:
                    if self.player == "Player 1":
                        node.points = 1
                    if self.player == "Player 2":
                        node.points = -1
                elif match_result == -1:
                    if self.player == "Player 1":
                        node.points = -1
                    if self.player == "Player 2":
                        node.points = 1
                else:
                    node.points = 0

            return node.points

        recurse(self._root)

    def best_move(self):
        """
        Returns the best possible move starting from the root
        """
        maximum = self._root.children[0]

        for child in self._root.children:
            if child.points >= maximum.points:
                maximum = child

        return maximum.board.last_turn[1]
