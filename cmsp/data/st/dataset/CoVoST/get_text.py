import os
import csv
import yaml
import pandas as pd
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-d", type=str, required=True,
                        help="path to dataset file for the Xx language")
    parser.add_argument("--src-lang", "-s", type=str, required=True,
                        help="source language code")
    parser.add_argument("--tgt-lang", "-t", type=str, required=True,
                        help="target language code")
    parser.add_argument("--mode", "-v", type=str, choices=["train", "dev", "test"],
                        required=True, help="text splits")
    return parser.parse_args()


def load_df_from_tsv(p):
    return pd.read_csv(p, sep="\t", header=0, encoding="utf-8", escapechar="\\", quoting=csv.QUOTE_NONE,
                       na_filter=False)


def get_lt(p, lt1, lt2):
    tsv = load_df_from_tsv(p)
    print(f"该tsv共{len(tsv)}条")

    for sens, trans in zip(tsv["sentence"], tsv["translation"]):
        lt1.append(sens)
        lt2.append(trans)

    return lt1, lt2


def write2text(p, lang1, lang2, mode):
    sens, trans = [], []

    ftsv = p + "covost_v2." + lang1 + "_" + lang2 + "." + mode + ".tsv"
    wt1 = p + "/text/" + mode + "." + lang1
    wt2 = p + "/text/" + mode + "." + lang2

    with open(wt1, "w") as fwt1, open(wt2, "w") as fwt2:
        sens, trans = get_lt(ftsv, sens, trans)
        print(f"共{len(sens)}条sentences和translation")

        for sentence, translation in zip(sens, trans):
            fwt1.write(sentence + "\n")
            fwt2.write(translation + "\n")

        print(f"text文件写入成功")


def main():
    args = get_args()

    path, src, tgt, mode = args.path, args.src_lang, args.tgt_lang, args.mode
    assert src != tgt and "en" in {src, tgt}

    if not os.path.exists(path + "text"):
        os.makedirs(path + "text")

    write2text(path, src, tgt, mode)


if __name__ == "__main__":
    main()
