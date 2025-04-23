#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple, Any, Union

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(sys.path[0])))))

import torch
import numpy as np
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
from fairseq.examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
)
from torch.utils.data import Dataset
from tqdm import tqdm

log = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_text", "tgt_text", "speaker"]


def convert_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    to_mono: bool = False,
    to_sample_rate: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], int]:

    is_np_input = isinstance(waveform, np.ndarray)
    _waveform = torch.from_numpy(waveform) if is_np_input else waveform
    if to_sample_rate is not None and to_sample_rate != sample_rate:
        resample = T.Resample(orig_freq=sample_rate, new_freq=to_sample_rate)
        _waveform = resample(_waveform)
    if to_mono and waveform.shape[0] > 1:
        mono_transform = T.StereoToMono()
        _waveform = mono_transform(_waveform)
    if is_np_input:
        _waveform = _waveform.numpy()
    return _waveform, to_sample_rate


class CoVoST(Dataset):
    """Create a Dataset for CoVoST (https://github.com/facebookresearch/covost).

    Args:
        root (str): root path to the dataset and generated manifests/features
        source_language (str): source (audio) language
        target_language (str, optional): target (text) language,
        None for no translation (default: None)
        version (int, optional): CoVoST version. (default: 2)
        download (bool, optional): Whether to download the dataset if it is not
        found at root path. (default: ``False``).
    """

    COVOST_URL_TEMPLATE = (
        "covost_v2.{src_lang}_{tgt_lang}.tsv.tar.gz"
    )

    VERSIONS = {2}
    SPLITS = ["train", "dev", "test"]

    XX_EN_LANGUAGES = {
        1: ["fr", "de", "nl", "ru", "es", "it", "tr", "fa", "sv-SE", "mn", "zh-CN"],
        2: [
            "fr",
            "de",
            "es",
            "ca",
            "it",
            "ru",
            "zh-CN",
            "pt",
            "fa",
            "et",
            "mn",
            "nl",
            "tr",
            "ar",
            "sv-SE",
            "lv",
            "sl",
            "ta",
            "ja",
            "id",
            "cy",
        ],
    }
    EN_XX_LANGUAGES = {
        1: [],
        2: [
            "de",
            "tr",
            "fa",
            "sv-SE",
            "mn",
            "zh-CN",
            "cy",
            "ca",
            "sl",
            "et",
            "id",
            "ar",
            "ta",
            "lv",
            "ja",
        ],
    }

    def __init__(
            self,
            root: str,
            split: str,
            source_language: str,
            target_language: Optional[str] = None,
            version: int = 2,
    ) -> None:
        assert version in self.VERSIONS and split in self.SPLITS
        assert source_language is not None
        self.no_translation = target_language is None
        if not self.no_translation:
            assert "en" in {source_language, target_language}
            if source_language == "en":
                assert target_language in self.EN_XX_LANGUAGES[version]
            else:
                assert source_language in self.XX_EN_LANGUAGES[version]
        else:
            # Hack here so that we can get "split" column from CoVoST TSV.
            # Note that we use CoVoST train split for ASR which is an extension
            # to Common Voice train split.
            target_language = "de" if source_language == "en" else "en"

        self.root: Path = Path(root)

        cv_tsv_path = self.root / "validated.tsv"
        assert cv_tsv_path.is_file()

        covost_url = self.COVOST_URL_TEMPLATE.format(
            src_lang=source_language, tgt_lang=target_language
        )
        # covost_archive = self.root / Path(covost_url).name
        # if not covost_archive.is_file():
        #     download_url(covost_url, self.root.as_posix(), hash_value=None)
        # extract_archive(covost_archive.as_posix())

        cv_tsv = load_df_from_tsv(cv_tsv_path)
        covost_tsv = load_df_from_tsv(
            self.root / Path(covost_url).name.replace(".tar.gz", "")
        )
        df = pd.merge(
            left=cv_tsv[["path", "sentence", "client_id"]],
            right=covost_tsv[["path", "translation", "split"]],
            how="inner",
            on="path",
        )
        if split == "train":
            df = df[(df["split"] == split) | (df["split"] == f"{split}_covost")]
        else:
            df = df[df["split"] == split]
        data = df.to_dict(orient="index").items()
        data = [v for k, v in sorted(data, key=lambda x: x[0])]
        self.data = []
        for e in data:
            try:
                path = self.root / "clips" / e["path"]
                _ = torchaudio.info(path.as_posix())
                self.data.append(e)
            except RuntimeError:
                pass

    def __getitem__(
            self, n: int
    ) -> tuple[Any, Any, Any, Optional[Any], Any, Any]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, sentence, translation, speaker_id,
            sample_id)``
        """
        data = self.data[n]
        path = self.root / "clips" / data["path"]
        waveform, sample_rate = torchaudio.load(path, format="mp3")
        sentence = data["sentence"]
        translation = None if self.no_translation else data["translation"]
        speaker_id = data["client_id"]
        _id = data["path"].replace(".mp3", "")
        return waveform, sample_rate, sentence, translation, speaker_id, _id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    audio_type = "raw" if args.use_audio_input else "fb"
    root = Path(args.data_root).absolute() / args.src_lang
    audio_root = Path(args.spm_data).absolute() / f"{args.src_lang}-{args.tgt_lang}"
    spm_root = Path(args.spm_info).absolute() / f"{args.src_lang}-{args.tgt_lang}"
    audio_root.mkdir(exist_ok=True)
    spm_root.mkdir(exist_ok=True)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} does not exist")

    # Extract features
    feature_root = root / ("flac" if args.use_audio_input else "fbank80")
    feature_root.mkdir(exist_ok=True)
    for split in CoVoST.SPLITS:
        print(f"Fetching split {split}...")
        dataset = CoVoST(root, split, args.src_lang, args.tgt_lang)

        if args.use_audio_input:
            print("Converting audios...")
            for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
                tgt_sample_rate = 16_000
                _wavform, _ = convert_waveform(
                    waveform, sample_rate, to_mono=True,
                    to_sample_rate=tgt_sample_rate
                )
                sf.write(
                    (feature_root / f"{utt_id}.flac").as_posix(),
                    _wavform.T.numpy(), tgt_sample_rate
                )
        else:
            print("Extracting log mel filter bank features...")
            for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
                extract_fbank_features(
                    waveform, sample_rate, feature_root / f"{utt_id}.npy"
                )

    # Pack features into ZIP
    zip_path = audio_root / f"{feature_root.name}.zip"
    print("ZIPing features...")
    create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(zip_path, is_audio=args.use_audio_input)

    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    task = f"asr_{args.src_lang}"
    if args.tgt_lang is not None:
        task = f"st_{args.src_lang}_{args.tgt_lang}"
    for split in CoVoST.SPLITS:
        is_train_split = split.startswith("train")
        if is_train_split:
            MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_text", "tgt_text", "speaker"]
        else:
            MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = CoVoST(root, split, args.src_lang, args.tgt_lang)
        for _, _, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
            manifest["id"].append(utt_id)
            manifest["audio"].append(audio_paths[utt_id])
            manifest["n_frames"].append(audio_lengths[utt_id])
            if is_train_split:
                manifest["src_text"].append(src_utt)
            manifest["tgt_text"].append(src_utt if args.tgt_lang is None else tgt_utt)
            manifest["speaker"].append(speaker_id)
        if is_train_split:
            train_text.extend(manifest["src_text"])
            train_text.extend(manifest["tgt_text"])
        df = pd.DataFrame.from_dict(manifest)
        if args.use_audio_input:
            df = filter_manifest_df(df, is_train_split=is_train_split, min_n_frames=args.min_n_frames,
                                    max_n_frames=args.max_n_frames)
        else:
            df = filter_manifest_df(df, is_train_split=is_train_split)
        save_df_to_tsv(df, root / f"{split}_{task}_{audio_type}.tsv")

    # Generate vocab
    vocab_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size_str}_{task}"
    spm_model = spm_filename_prefix + ".model"
    spm_dict = spm_filename_prefix + ".txt"

    with NamedTemporaryFile(mode="w") as f:
        for t in train_text:
            f.write(t + "\n")
        gen_vocab(
            Path(f.name),
            spm_root / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size
        )

    # Generate config YAML
    if args.use_audio_input:
        gen_config_yaml(
            spm_root,
            spm_filename=spm_model,
            vocab_name=spm_dict,
            yaml_filename=f"config_{task}.yaml",
            specaugment_policy=None,
            extra={"use_audio_input": True}
        )
    else:
        gen_config_yaml(
            spm_root,
            spm_filename=spm_model,
            yaml_filename=f"config_{task}.yaml",
            specaugment_policy="lb",
        )

    Clean up
    shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", "-d", required=True, type=str,
        help="data root with sub-folders for each language <root>/<src_lang>"
    )
    parser.add_argument("--spm-info", "-i", required=True, type=str)
    parser.add_argument("--spm-data", "-m", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=1000, type=int)
    parser.add_argument("--src-lang", "-s", required=True, type=str)
    parser.add_argument("--tgt-lang", "-t", type=str)
    parser.add_argument("--min_n_frames", default=1000, type=int)
    parser.add_argument("--max_n_frames", default=480000, type=int)
    parser.add_argument("--use-audio-input", action="store_true")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
