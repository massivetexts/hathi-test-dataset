from SRP import Vector_file
import numpy as np

seen = set()
with Vector_file("glove.bin", mode = "w", dims = 300) as fout:
    for i, line in enumerate(open("glove.840B.300d.txt")):
        if i % 10000 == 0:
            print(i)
        word, data = line.split(" ", 1)
        if word in seen:
            print("two copies of {}".format(word))
            continue
        seen.add(word)
        data = np.fromstring(data, '<f4', sep=' ')
        fout.add_row(word, data)
        if i > 200000:
            break
        
