import itertools, math

words = ["BE","OAK","ROOM","IDEAL","SCHOOL","KITCHEN","OVERCOAT","REVOLVING","DEMOCRATIC","ENTERTAINER","MATHEMATICAL","SPORTSMANSHIP","KINDERGARTENER","INTERNATIONALLY"]
lengths = {w:len(w) for w in words}

# EVOLVE-BLOCK-START
def your_function():
    # Group words by their lengths for fast lookup
    from collections import defaultdict
    length_to_words = defaultdict(list)
    for w in words:
        length_to_words[len(w)].append(w)
    lengths_set = set(length_to_words.keys())
    max_len = max(lengths_set)

    # Mathematical reduction:
    # let r = a/b (must be integer). Then a = r*b, c = r^3, d = r^2*b.
    # So we only need to try integer r and base length b.
    solutions = []
    for b in lengths_set:
        max_r = max_len // b
        for r in range(1, max_r + 1):
            a = r * b
            d = r * r * b
            c = r ** 3
            # all lengths must exist in the available set
            if a in lengths_set and d in lengths_set and c in lengths_set:
                # form combinations of actual words of those lengths, ensuring distinct words
                for wa in length_to_words[a]:
                    for wb in length_to_words[b]:
                        for wc in length_to_words[c]:
                            for wd in length_to_words[d]:
                                if len({wa, wb, wc, wd}) == 4:
                                    solutions.append((wa, wb, wc, wd, (a, b, c, d)))
    # canonicalize and print
    solutions = sorted(set(solutions))
    print(solutions)
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    your_function()
