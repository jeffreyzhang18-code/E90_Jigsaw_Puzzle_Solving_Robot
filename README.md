Repository for E90 Project - Jigsaw Puzzle Solving Robot by Jeffrey Zhang, Swarthmore College Class of 2025

README:

puzzle_1 contains images of scan of front of puzzle piece

puzzle_1_flipped contains images of scan of back puzzle piece, and individual pieces

imagecombiner.py combines multiple images into a single image horizontally

imagecropper.py crops images

movementselector.py provides Python script for sending G-code into machine, creates pop up windown, when you click on window CNC mill moves to location

puzzle_piece_generator.py separates piece into individual images, and finds best matches between edges

solved_puzzle_generator_edges_then_corners.py creates image of finalized puzzle by placing edges first, then center

solved_puzzle_generator_greedy.py creates image of finalized puzzle by placing pieces as soon as adjacent piece is place

solved_puzzle_generator_improved_greedy.py creates image of finalized puzzle by placing pieces based on all adjacent pieces and placing in steps where pieces placed are piece that are adjacent to previously placed pieces.
