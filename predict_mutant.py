from model import TemPL
import torch
from tqdm import tqdm
from utils import read_seq_from_fasta
import pandas as pd
import sys
from scipy.stats import spearmanr
from argparse import ArgumentParser
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def compute_probabilities_over_sequence(model: TemPL, wt_sequence_input_ids):
    probabilities = []
    for i in tqdm(range(0, wt_sequence_input_ids.size(1))):
        masked_input_ids = wt_sequence_input_ids.clone()
        masked_input_ids[0, i] = model.config.mask_token_id
        logits = model(masked_input_ids, task="mask_prediction")
        probabilities.append(torch.log_softmax(logits, dim=-1)[:, i])
    probabilities = torch.cat(probabilities, dim=0).unsqueeze(0)
    return probabilities


@torch.no_grad()
def score_mutants(model, tokenizer, wt_sequence, mutants):
    wt_input_ids = tokenizer(wt_sequence, return_tensors="pt")["input_ids"].to(device)
    probabilities = compute_probabilities_over_sequence(model, wt_input_ids)
    for mutant in mutants:
        mutant_score = []
        for m in mutant.split(";"):
            if m.lower() == "wt":
                wt, idx, mt = wt_sequence[0], 0, wt_sequence[0]
            else:
                wt, idx, mt = m[0], int(m[1:-1]) - 1, m[-1]
            assert wt_sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
            wt_id, mt_id = tokenizer.get_vocab()[wt], tokenizer.get_vocab()[mt]
            score = probabilities[0, 1 + idx, mt_id] - probabilities[0, 1 + idx, wt_id]
            mutant_score.append(score.item())
        mutant_score = sum(mutant_score) / len(mutant_score)
        yield mutant_score


@torch.no_grad()
def score_file(model, tokenizer, fasta_file, mutant_file, compute_spearman=True):
    wt_sequence = read_seq_from_fasta(fasta_file)
    df = pd.read_table(mutant_file)
    if len(df['mutant']) != len(df['mutant'].drop_duplicates()):
        # 使用标准错误输出
        duplicate_num = len(df['mutant']) - len(df['mutant'].drop_duplicates())
        duplicate_prop = duplicate_num / len(df['mutant'])
        sys.stderr.write(
            f"Warning: duplicate mutants found in {mutant_file}. "
            f"Duplicate number : {duplicate_num} : Prop : {duplicate_prop}\n"
        )
        # 去重
        df = df.drop_duplicates(subset=['mutant'], keep='first')
    df['prediction'] = list(score_mutants(model, tokenizer, wt_sequence, df['mutant']))
    return df


def main(args):
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = TemPL.load(args.model_name)
    model.to(device)
    df = score_file(model, tokenizer, args.fasta, args.mutant)
    if args.compute_spearman:
        assert 'score' in df.columns, "score column not found in the mutant file"
        spearman = spearmanr(df['prediction'], df['score'])
        print(f"Spearman correlation: {spearman.correlation:.3f} (p-value: {spearman.pvalue:.3f})")
    if args.output is not None:
        df.to_csv(args.output, sep="\t", index=False)


if __name__ == '__main__':
    def create_parser():
        psr = ArgumentParser()
        psr.add_argument("--model_name", type=str,
                         default="templ-tm-fine-tuning",
                         choices=["templ-tm-fine-tuning", "templ-base"],
                         help="model name")
        psr.add_argument("--fasta", type=str, required=True, help="fasta file, only one sequence")
        psr.add_argument("--mutant", type=str,
                         required=True, help="mutant file, tsv format, 1 or 2 columns: mutant score(optional)")
        psr.add_argument("--compute_spearman", action="store_true", default=False,
                         help="compute spearman correlation between prediction and score")
        psr.add_argument("--output", type=str, default=None, help="output file")
        return psr


    parser = create_parser()
    cli_args = parser.parse_args()
    main(cli_args)
