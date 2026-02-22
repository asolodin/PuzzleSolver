import itertools, math

words = ["BE","OAK","ROOM","IDEAL","SCHOOL","KITCHEN","OVERCOAT","REVOLVING","DEMOCRATIC","ENTERTAINER","MATHEMATICAL","SPORTSMANSHIP","KINDERGARTENER","INTERNATIONALLY"]
lengths = {w:len(w) for w in words}

# EVOLVE-BLOCK-START
def your_function():
    # map lengths -> list of words (fast lookup by length)
    lengths_map = {}
    for w in words:
        lengths_map.setdefault(len(w), []).append(w)

    solutions = []
    max_len = max(lengths_map.keys())

    # algebraic reduction:
    # from a^2 = b*d and a*d = b^2*c we get c = (a/b)^3 and a = k*b, c = k^3, d = k^2*b
    # so k must be an integer with k^3 <= max_len. k=1 gives c=1 (too small here), so start at 2.
    max_k = 1
    while (max_k + 1) ** 3 <= max_len:
        max_k += 1

    for k in range(2, max_k + 1):
        c_len = k ** 3
        if c_len not in lengths_map:
            continue
        # iterate possible base length b
        for b_len in list(lengths_map.keys()):
            a_len = k * b_len
            d_len = (k * k) * b_len
            if a_len in lengths_map and d_len in lengths_map:
                # produce all distinct-word combinations of the required lengths
                for wa in lengths_map[a_len]:
                    for wb in lengths_map[b_len]:
                        for wc in lengths_map[c_len]:
                            for wd in lengths_map[d_len]:
                                if len({wa, wb, wc, wd}) == 4:
                                    solutions.append((wa, wb, wc, wd, (a_len, b_len, c_len, d_len)))
    print(solutions)
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    your_function()
