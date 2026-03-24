
import torch
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from itertools import combinations
import re
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.model_selection import train_test_split

def generalized_js_distance(distributions, epsilon=1e-10):
    """
    Computes generalized Jensen-Shannon **distance** for multiple distributions.
    
    Args:
        distributions: list of arrays (PDFs evaluated over same xs)
        epsilon: small value added for numerical stability
    
    Returns:
        Generalized JS distance (float)
    """
    k = len(distributions)
    dists = [dist + epsilon for dist in distributions]
    dists = [d / np.sum(d) for d in dists]
    mixture = np.mean(dists, axis=0)

    def entropy(p):
        return -np.sum(p * np.log2(p))

    js = entropy(mixture) - np.mean([entropy(p) for p in dists])
    return np.sqrt(js) / np.sqrt(np.log2(k))  # Normalized to [0, 1]

def is_match(a, b):
    a_clean = a.replace(" ", "").lower()
    b_clean = b.replace(" ", "").lower()
    return a_clean == b_clean

def combine_defaultdicts(*dicts):
    combined = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            combined[key].extend(value)
    return combined

def tokenize_prompt_dataset(
    texts: list[str],
    tokenize,
    pad_token_id,
    in_context_prompt: str,        # your in‐context prompt string
    end_of_text: str,              # your end‐of‐text marker string
    context_length: int = 512
) -> dict[str, torch.Tensor]:
    prompt_input_ids = []
    prompt_attention_masks = []

    # Tokenize static parts once
    prompt_tokens = tokenize(in_context_prompt)[0]
    end_of_text_tokens = tokenize(end_of_text)[0][1:]  # drop leading bos/eos as needed

    for text in tqdm(texts,desc="tokenizing prompts",  total=len(texts)):

        # tokenize text and label
        text_tokens   = tokenize(text)[0][1:]

        # compute how much padding is needed
        unused_len = (
            context_length
            - len(prompt_tokens)
            - len(text_tokens)
            - len(end_of_text_tokens)
        )

        # if too long, truncate from the left of text_tokens
        if unused_len < 0:
            text_tokens = text_tokens[unused_len * -1:]
            unused_len = 0

        # build input_ids and attention_mask
        pad = torch.full([unused_len], pad_token_id, dtype=torch.long)
        input_ids = torch.cat([pad, prompt_tokens, text_tokens, end_of_text_tokens])
        attention_mask = input_ids.ne(pad_token_id).to(torch.int32)

        prompt_input_ids.append(input_ids)
        prompt_attention_masks.append(attention_mask)

    return {
        "input_ids":      torch.stack(prompt_input_ids),
        "attention_mask": torch.stack(prompt_attention_masks)
    }
    
def tokenize_analyzing_dataset(
    texts: list[str],
    tokenize,
    pad_token_id,
    context_length: int = 512,
    truncation_size = "right",
    padding_side = "right"
) -> dict[str, torch.Tensor]:
    prompt_input_ids = []
    prompt_attention_masks = []

    for text in tqdm(texts,desc="tokenizing prompts",  total=len(texts)):

        # tokenize text and label
        text_tokens   = tokenize(text)[0]

        # compute how much padding is needed
        unused_len = (
            context_length
            - len(text_tokens)
        )

        if truncation_size == "right":
            # if too long, truncate from the right of text_tokens
            if unused_len < 0:
                text_tokens = text_tokens[:context_length]
                unused_len = 0
        elif truncation_size == "left":
            if unused_len < 0:
                text_tokens = text_tokens[-context_length:]
                unused_len = 0
            

        # build input_ids and attention_mask
        pad = torch.full([unused_len], pad_token_id, dtype=torch.long)
        if padding_side == "right":
            input_ids = torch.cat([text_tokens, pad])
        else:
            input_ids = torch.cat([pad, text_tokens])
            
        attention_mask = input_ids.ne(pad_token_id).to(torch.int32)

        prompt_input_ids.append(input_ids)
        prompt_attention_masks.append(attention_mask)

    return {
        "input_ids":      torch.stack(prompt_input_ids),
        "attention_mask": torch.stack(prompt_attention_masks)
    }




def detok(tokens):
    s = " ".join(tokens)
    s = re.sub(r"\s+([,.;:!?%])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\[\s+", "[", s)
    s = re.sub(r"\s+\]", "]", s)
    s = re.sub(r'"\s+', '"', s)
    s = re.sub(r'\s+"', '"', s)
    s = re.sub(r"\s+'", "'", s)
    s = re.sub(r"'\s+", "'", s)
    s = re.sub(r"\s+(/)", r"\1", s)
    s = re.sub(r"(/)\s+", r"\1", s)
    return s

def build_triplets(xml_file: Path):
    # Parse XML
    tree = ET.parse(xml_file)
    root = tree.getroot()

    records = []
    # Iterate sentences with a progress bar
    for sentence in tqdm(root.findall(".//sentence"), desc="Processing sentences"):
        # First pass: collect token texts to reconstruct sentence
        token_texts = []
        for child in sentence:
            if child.tag in ("wf", "instance"):
                tok_text = (child.text or "").strip()
                token_texts.append(tok_text)
        sent_text = detok(token_texts)

        # Second pass: collect eligible tokens (wf + instance)
        for child in sentence:
            if child.tag in ("wf", "instance"):
                tok_text = (child.text or "").strip()
                if not tok_text:
                    continue
                pos = (child.attrib.get("pos") or "").upper()
                if pos in {"NOUN", "VERB", "ADJ", "ADV"}:
                    records.append({"text": sent_text, "word": tok_text, "pos": pos})

    # Build aligned columns; sanity check
    data = {
        "text": [r["text"] for r in records],
        "word": [r["word"] for r in records],
        "pos":  [r["pos"]  for r in records],
    }
    assert len(data["text"]) == len(data["word"]) == len(data["pos"]), "Lists are not aligned!"
    return records

def split_stratified(df: pd.DataFrame, test_size=0.2, seed=42):
    """Use all data; preserve class proportions via stratification."""

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=df["pos"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def tokenize_prompt_word_dataset(
    texts: list[str],
    words: list[str],
    tokenize,
    pad_token_id,
    in_context_prompt: str,        # your in‐context prompt string
    end_of_text: str,              # your end‐of‐text marker string
    context_length: int = 512
) -> dict[str, torch.Tensor]:
    prompt_input_ids = []
    prompt_attention_masks = []

    # Tokenize static parts once
    prompt_tokens = tokenize(in_context_prompt)[0]
    end_of_text_tokens = tokenize(end_of_text)[0][1:]  # drop leading bos/eos as needed

    for text, word in tqdm(zip(texts, words),desc="tokenizing prompts",  total=len(texts)):

        # tokenize text and label
        text_tokens   = tokenize(text)[0][1:]
        word_tokens = tokenize(word)[0][1:]

        # compute how much padding is needed
        unused_len = (
            context_length
            - len(prompt_tokens)
            - len(text_tokens)
            - len(end_of_text_tokens)
            - len(word_tokens)
        )

        # if too long, truncate from the left of text_tokens
        if unused_len < 0:
            text_tokens = text_tokens[unused_len * -1:]
            unused_len = 0

        # build input_ids and attention_mask
        pad = torch.full([unused_len], pad_token_id, dtype=torch.long)
        input_ids = torch.cat([pad, prompt_tokens, text_tokens, end_of_text_tokens, word_tokens])
        attention_mask = input_ids.ne(pad_token_id).to(torch.int32)

        prompt_input_ids.append(input_ids)
        prompt_attention_masks.append(attention_mask)

    return {
        "input_ids":      torch.stack(prompt_input_ids),
        "attention_mask": torch.stack(prompt_attention_masks)
    }
    
def convert_few_nerd_to_dict(dataset_split):
    coarse_names = dataset_split.features["ner_tags"].feature.names
    O_ID = coarse_names.index("O")
    OTHER_ID = coarse_names.index("other")

    tag_key = "ner_tags"
    label_names = dataset_split.features[tag_key].feature.names

    out = {"text": [], "word": [], "ner": []}

    for ex in tqdm(dataset_split, desc="Processing examples"):
        # Filter out empty tokens (Few-NERD sometimes has '')
        tokens0 = ex["tokens"]
        tags0   = ex[tag_key]
        tokens = []
        tags   = []
        for t, y in zip(tokens0, tags0):
            if t is None or t == "":
                continue
            tokens.append(t)
            tags.append(y)

        if not tokens:
            continue

        sent_text = detok(tokens)

        # group contiguous same-tag (non-O) spans -> full entity surface
        i = 0
        n = len(tokens)
        while i < n:
            tag = tags[i]
            if tag == O_ID or tag == OTHER_ID:
                i += 1
                continue
            j = i + 1
            while j < n and tags[j] == tag:
                j += 1
            span_surface = detok(tokens[i:j])
            out["text"].append(sent_text)
            out["word"].append(span_surface)
            out["ner"].append(label_names[tag])
            i = j
    return pd.DataFrame(out)