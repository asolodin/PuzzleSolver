import itertools, math

words = ["BE","OAK","ROOM","IDEAL","SCHOOL","KITCHEN","OVERCOAT","REVOLVING","DEMOCRATIC","ENTERTAINER","MATHEMATICAL","SPORTSMANSHIP","KINDERGARTENER","INTERNATIONALLY"]
lengths = {w:len(w) for w in words}

# EVOLVE-BLOCK-START
def your_function():
    # available (word, length) pairs
    pairs = [(w, len(w)) for w in words]
    solutions = []
    for (wa, a), (wb, b), (wc, c), (wd, d) in itertools.permutations(pairs, 4):
        if a * a == b * d and a * d == (b * b * c):
            solutions.append((wa, wb, wc, wd, (a, b, c, d)))
    print(solutions[:20])
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    your_function()
