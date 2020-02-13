from pathlib import Path
from SRP import Vector_file

glove = Path("data_outputs/glove.bin")
my_SRP = Path("data_outputs/SRP.bin")

completed = Path("data_outputs/completed.csv")

# Write completed files.
with completed.open("a") as fout:
    for fin in Path("data_outputs").glob("already_completed*.csv"):
        for line in fin.open():
            fout.write(line)
        fin.unlink()

done = set([l.rstrip() for l in completed.open('r')])
new_entries = []

for fname, match, d in [(glove, "Glove", 300), (my_SRP, "SRP", 640)]:
    if not Path.exists(fname):
        mode = "w"
    else:
        mode = "a"
    with Vector_file(fname, dims = d, mode = mode) as fout:
        for tempfile in Path("data_outputs").glob(f"*_{match}_chunks.bin"):
            z = Vector_file(tempfile, mode = 'r')
            last = None
            for i, (id, row) in enumerate(z):
                htid, rest = id.split("-", 1)
                if z.vocab_size - i < 100:
                    # Don't do the last 100 items.
                    break
                last = htid
                if not htid in done:
                    done.add(htid)
                    new_entries.append(htid)
                fout.add_row(id, row)
            tempfile.unlink()


print(f"{len(new_entries)} new entries")

with completed.open("a") as fout:
    for entry in new_entries:
        fout.write(entry + "\n")
