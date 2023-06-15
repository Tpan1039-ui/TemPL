from typing import List
import pandas as pd
from Bio import SeqIO


def read_seq_from_fasta(fasta_file_path):
    fasta = SeqIO.read(fasta_file_path, 'fasta')
    sequence = str(fasta.seq)
    return sequence


def full_sequence(origin_sequence, raw_mutant):
    list_mutants = raw_mutant.split(";")
    sequence = origin_sequence
    for raw_mut in list_mutants:
        to = raw_mut[-1]
        pos = int(raw_mut[1:-1]) - 1
        assert sequence[pos] == raw_mut[
            0], "the original sequence is different to that in the mutant file in resid %d" % (pos + 1)
        sequence = sequence[:pos] + to + sequence[pos + 1:]
    return sequence


def read_mutant_seqs(origin_sequence: str, tsv_path) -> List[str]:
    if isinstance(tsv_path, list):
        sequences = []
        for each in tsv_path:
            sequences.extend(read_mutant_seqs(origin_sequence, each))
        return sequences
    table = pd.read_table(tsv_path)
    sequences = []
    for raw_mutant in table['mutant']:
        mutant_sequence = full_sequence(origin_sequence, raw_mutant)
        sequences.append(mutant_sequence)
    return sequences
