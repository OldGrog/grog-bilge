import numpy as np
import time
import cv2
from cv2 import matchTemplate as cv2m
import cProfile
from collections import deque


# ## Random 12x6 numpy array filled with objects valued 0-6

A = np.array([[0, 1, 1, 0, 1, 0],
              [0, 3, 4, 6, 0, 2],
              [1, 0, 4, 5, 0, 0],
              [0, 1, 0, 3, 3, 0],
              [0, 2, 4, 4, 0, 5],
              [2, 2, 3, 1, 0, 5],
              [3, 6, 5, 0, 5, 6],
              [1, 2, 0, 2, 0, 6],
              [0, 0, 1, 1, 0, 0],
              [2, 4, 0, 5, 5, 3],
              [1, 0, 4, 0, 2, 2],
              [3, 3, 1, 2, 6, 6]], dtype=object)


# ## Checks via cv module whether a certain 1D or 2D array exists in another array.
# ## Use it ATM to check if theres a vertical or horizontal triple of the same piece in a board state.
# ## Input is 2 numpy arrays. Output is a list of (row, column) tuples.

def match_in_fieldmatrix(boardstate, match):
    m = cv2m(boardstate.astype('uint8'), match.astype('uint8'), cv2.TM_SQDIFF)

    r, c = np.where(m == 0)

    if len(r) == 0:
        return None
    else:
        return list(zip(r, c))


# ## Using the aforementioned cv method function, loop through all possible triples of elements 0-7.
# ## and find an appropriate match and give it a score of 3, checking no further ATM.
# ## Input is just the board state, output is 0 if no match, 3 for match.

def evaluate_combo(boardstate):
    for i in range(7):

        c = np.full((1, 3), i)

        if match_in_fieldmatrix(boardstate, c) or match_in_fieldmatrix(boardstate, c.T):
            return 3
    return 0


# ## Find all child-nodes of parent node.
# ## Input is a parent-node boardstate, output is a generator iterable of all child-nodes.
# ## Nodes with adjacent elements that are the same are omitted from the output iterable.

def elementswap_getchildren(matrix):

    height, width = matrix.shape

    for i, j in [(i, j) for i in range(height) for j in range(width - 1) if matrix[i, j] != matrix[i, j + 1]]:

        child = matrix.copy()

        child[i, j], child[i, j + 1] = child[i, j + 1], child[i, j]

        yield child


# ## The primary breadth-first search function. Visited is the set of all nodes that
# ## we have visited and generated its child-nodes. Queue is a dequeue structure, which
# ## we need, to keep a queue of the nodes and their child nodes in an appropriate bfs order.
# ## If a current node is unique, we add it to the visited set, then we check whether we are not
# ## in the last depth. If the node boardstate has a non-0 score evaluation, we will not look at its
# ## child node(s). Otherwise we will use the generator iterables to append each child-node to the queue.
# ## After that using some temporary variable bullshit I basically store the amount of unique boardstates
# ## on each depth to predict when the depths change. Not sure if that can be resolved to better/pythonic code.

def bfs(initial, depth):
    visited = set()

    queue = deque([initial])

    i, j, k, toggle = 0, 0, 0, 0

    while queue:
        node = queue.popleft()

        node_tuple = tuple(map(tuple, node))

        if node_tuple not in visited:

            visited.add(node_tuple)

            if depth != 0 and evaluate_combo(node) == 0:  # and visited[node_tuple] is None:
                for child in elementswap_getchildren(node):
                    queue.append(child)
                    i += 1

            if toggle == 0:

                k = i
                depth -= 1
                toggle = 1
        j += 1

        if j == k:
            k = i
            if depth != 0:
                depth -= 1

    return visited


# ## Lastly calling out the function via bfs, with its arguments: initial numpy array boardstate/node
# ## and the searchable depth of interest. The end goal is to use a dictionary instead of set in bfs.
# ## Key would be a board state, value would be the boardstate evaluation divided by the required moves/depth.
# ## Using the dictionary, it would be rather straightfoward to find node with the highest value.

if __name__ == "__main__":
    start = time.time()
    results = bfs(A, 5)
    end = time.time()

    print('Visited', len(results), 'positions')
    print('This took', end - start, 'seconds')
