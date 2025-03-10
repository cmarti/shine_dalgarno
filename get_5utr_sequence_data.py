from os.path import join

import pandas as pd

from pysam import Fastafile


def reverse_complement(seq):
    COMPLEMENT = {"A": "U", "T": "A", "U": "A", "G": "C", "C": "G"}
    return "".join(COMPLEMENT.get(x, x) for x in seq[::-1])


def get_seq(genome, chrom, start, end, strand):
    seq = genome.fetch(chrom, start, end)
    if strand == "-":
        seq = reverse_complement(seq)

    return seq.upper()


def read_gff(gtf_fpath):
    with open(gtf_fpath) as fhand:
        for line in fhand:
            if line.startswith("#"):
                continue
            items = line.strip().split("\t")
            if items[2] != "CDS":
                continue
            chrom, strand = items[0], items[6]
            pos = items[3] if strand == "+" else int(items[4]) - 3
            start, end = int(pos) - 1, int(pos) + 3

            attrs = dict(x.strip().split("=") for x in items[-1].split(";"))
            gene_id = attrs["gene"] if "gene" in attrs else attrs["ID"]
            yield (chrom, start, end, strand, gene_id)


def read_gtf(gtf_fpath):
    with open(gtf_fpath) as fhand:
        for line in fhand:
            if line.startswith("#"):
                continue
            items = line.strip().split("\t")
            if items[2] != "start_codon":
                continue
            chrom, strand = items[0], items[6]
            start, end = int(items[3]) - 1, int(items[4])
            attrs = dict(
                x.strip().split(" ") for x in items[-1].strip('";').split(";")
            )
            gene_id = attrs["gene_id"].strip('"')
            yield (chrom, start, end, strand, gene_id)


def read_annotation(fpath):
    if fpath.endswith(".gtf"):
        return read_gtf(fpath)
    elif fpath.endswith(".gff"):
        return read_gff(fpath)
    else:
        raise ValueError("Unknown annotation file format")


def get_upstream_coords(start, end, flanking_bases):
    start, end = start - flanking_bases, end + flanking_bases
    start = max(start, 0)
    return (start, end)


def get_SD_seqs(annotation, genome, seq_length, upstream_bases=20, sd_start=7):
    for chrom, start, end, strand, gene_id in annotation:
        start, end = get_upstream_coords(start, end, upstream_bases)
        seq = get_seq(genome, chrom, start, end, strand).replace("T", "U")
        sd_seq = seq[sd_start : sd_start + seq_length]
        start_codon = seq[upstream_bases : upstream_bases + 3]

        if len(seq) < 2 * upstream_bases + 3 or "N" in sd_seq:
            continue

        record = {
            "gene_id": gene_id,
            "start": start,
            "end": end,
            "strand": strand,
            "SD": sd_seq,
            "start_codon": start_codon,
            "background": seq,
        }
        yield (record)


def get_5UTR_seqs(annotation, genome, upstream_bases=20):
    for chrom, start, end, strand, gene_id in annotation:
        start, end = get_upstream_coords(start, end, upstream_bases)
        seq = get_seq(genome, chrom, start, end, strand).replace("T", "U")
        start_codon = seq[upstream_bases : upstream_bases + 3]

        if len(seq) < 2 * upstream_bases + 3:
            continue

        record = {
            "gene_id": gene_id,
            "start": start,
            "end": end,
            "strand": strand,
            "start_codon": start_codon,
            "background": seq,
        }
        yield (record)


if __name__ == "__main__":
    upstream_bases = 20

    # Extract E.coli sequences
    genomes = {
        "e_coli": "Escherichia_coli_gca_001263735.ASM126373v1.dna.toplevel.fa",
        "b_sub": "GCF_000009045.1_ASM904v1_genomic.fa",
    }
    annotations = {
        "e_coli": "Escherichia_coli_gca_001263735.ASM126373v1.51.gtf",
        "b_sub": "GCF_000009045.1_ASM904v1_genomic.gff",
    }

    for species in ["e_coli", "b_sub"]:
        print("Extracting 5'UTR sequences from {}".format(species))
        genome = Fastafile(join("genome", genomes[species]))
        annotation = read_annotation(join("genome", annotations[species]))

        gene_5utr = get_5UTR_seqs(
            annotation, genome, upstream_bases=upstream_bases
        )
        gene_5utr = pd.DataFrame(gene_5utr)
        gene_5utr.to_csv("processed/{}.gene_5utr.csv".format(species))
    print("Done")
