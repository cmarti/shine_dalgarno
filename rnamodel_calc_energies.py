import itertools


from subprocess import Popen, PIPE
from tqdm import tqdm
from csv import DictWriter


def group_lines(lines):
    group = []
    for line in lines:
        line = line.decode("utf-8").strip()
        if line.startswith(">"):
            if group:
                yield (group)
            group = [line]
        else:
            group.append(line)

    if group:
        yield (group)


def parse_group(group):
    seq = group[0].strip(">")
    deltaG = float(group[-1].split("=")[-1].strip())
    return {"seq": seq, "dG": deltaG}


if __name__ == "__main__":
    # rRNA_seq = "UCACCUCCUUA"
    aSD = "ACCUCCU"
    upstream_seq = "CCG"
    downstream_seq = "UGAG"

    seqs = ["".join(x) for x in itertools.product("ACGU", repeat=9)]

    print("Generating sequences...")
    print("\tUpstream sequence:{}".format(upstream_seq))
    print("\tDownstream sequence:{}".format(downstream_seq))

    fpath = "processed/SDaSD.fa"
    with open(fpath, "w") as fhand:
        for seq in seqs:
            seq_line = "{}&{}{}{}".format(
                aSD, upstream_seq, seq, downstream_seq
            )
            lines = ">{}\n{}\n"
            fhand.write(lines.format(seq, seq_line))

    print("Calculating energies...")
    cmd = ["RNAcofold", "-p0", "--noPS", fpath]
    p = Popen(cmd, stdout=PIPE)

    fpath = "processed/SDaSD.energies.csv"
    with open(fpath, "w") as fhand:
        writer = DictWriter(fhand, fieldnames=["seq", "dG"])
        writer.writeheader()
        for group in tqdm(group_lines(p.stdout)):
            record = parse_group(group)
            writer.writerow(record)
    print("Done")
