#!/usr/bin/env python3
"""
Sultanov Andriy
MIT License 2020
"""


class Node(object):
    """
    A Node class for the poly-tree
    """

    def __init__(self, board, points=0):
        self.board = board
        self.children = []
        self.points = points

    def __str__(self):
        return str(self.board)
