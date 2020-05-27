#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sultanov Andriy
"""
import cv2 as cv
import imutils
import numpy as np

# TODO: Abandoned project. The image somehow changes a lot without user, it is not possible to implement
#  proper user choice.
#  The game at this stage is a light proof of concept, and (was) a carcass for future implementations. The work
#  is discuntinued, and this app is not usable unless you just want to play with it
#  (it still works pretty fine with good lighting)


class GestureRecognizer:
    """
    Gesture Recognizer class
    """

    def __init__(self):
        self.background = None

        # Cells for Tic Tac Toe game
        cells = [(0.4 + (0.12 * (j - 1)), 0.2 + (0.09 * (i - 1)), 0.4 + (0.12 * j), 0.2 + (0.09 * i))
                 for j in range(3) for i in range(3)]

        self._choice(cells, 5, ["X", "O", "", "", "", ])

    def _choice(self, cells, time, tic_tac_toe_board=None):
        """
        Method creates a choice board from cells parameters, and returns the chosen cell after time passes.
        :param :cells: list
        :param :time: int
        :return:
        """
        # initialize weight for running average
        aWeight = 0.5

        # get the reference to the webcam
        camera = cv.VideoCapture(0)

        # initialize num of frames
        num_frames = 0

        # keep looping, until interrupted
        while True:
            # get the current frame
            (grabbed, frame) = camera.read()

            # resize the frame
            clone = frame.copy()
            clone = imutils.resize(clone, width=700)

            # flip the frame so that it is not the mirror view
            clone = cv.flip(clone, 1)

            # get the height and width of the frame
            height, width = clone.shape[:2]

            # Record the changes in pixels
            board = []

            # Check changes in every tic tac toe game cell
            for num, cell in enumerate(cells):
                top, right, bottom, left = cell

                # get the ROI
                roi = clone[int(height * top):int(height * bottom),
                            int(width * right):int(width * left)]

                # convert the roi to grayscale and blur it
                gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
                gray = cv.GaussianBlur(gray, (7, 7), 0)

                # to get the background, keep looking till a threshold is reached
                # so that our running average model gets calibrated
                if num_frames < 60:
                    self.run_avg(gray, aWeight)
                else:
                    # segment the item from the image
                    item = self.segment(gray)

                    # check whether this region is segmented
                    if item is not None:
                        # if yes, unpack the thresholded image and
                        # segmented region
                        thresholded, segmented = item

                        board.append(sum(sum(row) for row in thresholded))
                        # draw the segmented region and display the frame
                        # cv.drawContours(clone, [segmented + (int(width*right), int(height*top))], -1, (0, 0, 255))
                        # cv.imshow(f"Cell {num}", thresholded)
                    else:
                        board.append(0)

            # Checking if there is a cell with sudden movements
            # different from other cells
            if board:
                print(board)
                old_board = board.copy()
                # Removes cells' values that have changes too small to care about
                for num, cell in enumerate(board):
                    if cell < 100000 or cell < sum(old_board)/4.5:
                        board[num] = 0

                print(board)

                # Drawing the cells, the ones that are touched right now in a different color
                for num, cell in enumerate(cells):
                    top, right, bottom, left = cell
                    if max(board):
                        cv.rectangle(clone, (int(width * left), int(height * top)),
                                     (int(width * right), int(height * bottom)), (0, int(255 * (board[num] / (sum(board)/9))), 0), 2)
                    else:
                        cv.rectangle(clone, (int(width * left), int(height * top)),
                                     (int(width * right), int(height * bottom)), (0, 0, 0), 2)

            # Paint the current state of the tic tac toe board
            if tic_tac_toe_board is not None:
                for num, cell in enumerate(tic_tac_toe_board):
                    if cell == "X":
                        top, right, bottom, left = cells[num]
                        # Paint an "X" in the cell
                        center = ((int(width * left) + int(width * right)) // 2,
                                  (int(height * top) + int(height * bottom)) // 2)

                        points_1 = np.array([(center[0] + int(width * 0.03), center[1] + int(width * 0.025)),
                                             (center[0] + int(width * 0.025), center[1] + int(width * 0.03)),
                                             (center[0] - int(width * 0.03), center[1] - int(width * 0.025)),
                                             (center[0] - int(width * 0.025), center[1] - int(width * 0.03))]).astype(
                            np.int32)

                        points_2 = np.array([(center[0] + int(width * 0.03), center[1] - int(width * 0.025)),
                                             (center[0] + int(width * 0.025), center[1] - int(width * 0.03)),
                                             (center[0] - int(width * 0.03), center[1] + int(width * 0.025)),
                                             (center[0] - int(width * 0.025), center[1] + int(width * 0.03))]).astype(
                            np.int32)

                        cv.fillConvexPoly(clone, points_1, (0, 0, 0))
                        cv.fillConvexPoly(clone, points_2, (0, 0, 0))
                    elif cell == "O":
                        top, right, bottom, left = cells[num]
                        # Paint a circle in the cell
                        cv.circle(clone,
                                  ((int(width * left) + int(width * right)) // 2,
                                   (int(height * top) + int(height * bottom)) // 2),
                                  int(width * 0.035), (0, 0, 0), 2)

            # Display the frame
            cv.imshow("Video Feed", clone)

            # Observe the keypress by the user
            # if the user pressed "q", then stop looping
            keypress = cv.waitKey(1) & 0xFF
            if keypress == ord("q"):
                break

            # increment the number of frames
            num_frames += 1

        # free up memory
        camera.release()
        cv.destroyAllWindows()

    def run_avg(self, image, a_weight):
        """
        To find the running average over the background
        :param image:
        :param a_weight:
        :return:
        """
        # initialize the background
        if self.background is None:
            self.background = image.copy().astype("float")
            return

        # compute weighted average, accumulate it and update the background
        cv.accumulateWeighted(image, self.background, a_weight)

    def segment(self, image, threshold=25):
        """
        Segmenting the region from a video sequence
        :param image:
        :param threshold:
        :return:
        """
        # find the absolute difference between background and current frame
        diff = cv.absdiff(self.background.astype("uint8"), image)

        # threshold the diff image so that we get the foreground
        thresholded = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)[1]

        # get the contours in the thresholded image
        cnts = cv.findContours(thresholded.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

        # return None, if no contours detected
        if len(cnts) == 0:
            return
        else:
            # based on contour area, get the maximum contour which is the hand
            segmented = max(cnts, key=cv.contourArea)
            return thresholded, segmented


if __name__ == '__main__':
    recognizer = GestureRecognizer()
