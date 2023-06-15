from model import TemPL
import torch
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser
from transformers import AutoTokenizer
from Bio import SeqIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_sequence(model, tokenizer, sequence):
    input_ids = tokenizer(sequence, return_tensors="pt")["input_ids"].to(device)
    pred_ogt = model(input_ids, task="ogt_prediction").item()
    return pred_ogt


@torch.no_grad()
def score_file(model, tokenizer, fasta_file):
    records = list(SeqIO.parse(fasta_file, "fasta"))
    data = []
    for sequence in tqdm(records):
        pred_ogt = predict_sequence(model, tokenizer, str(sequence.seq))
        data.append({
            "sequence": str(sequence.seq),
            "predicted_ogt": pred_ogt
        })
    return pd.DataFrame(data)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = TemPL.load(args.model_name)
    model.to(device)
    df = score_file(model, tokenizer, args.fasta)
    if args.output is not None:
        df.to_csv(args.output, sep="\t", index=False)


if __name__ == '__main__':
    def create_parser():
        psr = ArgumentParser()
        psr.add_argument("--model_name", type=str,
                         default="templ-base",
                         choices=["templ-base"],
                         help="model name")
        psr.add_argument("--fasta", type=str, required=True, help="fasta file, only one sequence")
        psr.add_argument("--output", type=str, default=None, help="output file")
        return psr


    parser = create_parser()
    cli_args = parser.parse_args()
    main(cli_args)
