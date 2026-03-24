from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sae_lens import SAE, HookedSAETransformer
from tqdm import tqdm
import random
from datasets import load_dataset
import json
# import path
from pathlib import Path
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import pathlib
from sklearn.neighbors import KernelDensity
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utility import generalized_js_distance, build_triplets, split_stratified, tokenize_prompt_word_dataset, convert_few_nerd_to_dict


dataset_name = "ner" # use pos or ner
device = "cuda"
seed = 42
random.seed(seed)
model = "Llama-3.1-8B"  # Change this to the model you want to use (e.g., "gemma-2-2b", "Llama-3.1-8B")
if model == "gemma-2-2b":
    tokenizer =  AutoTokenizer.from_pretrained("google/gemma-2-2b")
    finetune_model = HookedSAETransformer.from_pretrained("gemma-2-2b", device=device, dtype=torch.float16)
    storage_path = Path("/u/siddique-d1/Moghis/storage")
    hook_name = "blocks.25.hook_resid_post"
    
    sae_id = "layer_25/width_16k/canonical"
    #sae_id = "layer_25/width_65k/average_l0_197"
    # sae_id = "layer_24/width_65k/average_l0_63"
    # sae_id = "layer_25/width_16k/canonical"
    # release = "gemma-scope-2b-pt-res-canonical"
    release = "gemma-scope-2b-pt-res-canonical"
    
elif model == "Llama-3.1-8B":
    tokenizer =  AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    hf_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    finetune_model = HookedSAETransformer.from_pretrained("meta-llama/Llama-3.1-8B", hf_model=hf_model, device=device, dtype=torch.float16)
    storage_path = Path("/u/siddique-d1/Moghis/storage/Llama")
    hook_name = "blocks.31.hook_resid_post"

    release = "llama_scope_r1_distill"
    sae_id = "l31r_800m_slimpajama"

combination_type = "last_token"  # Options: "last_token", "all_tokens_max"
#data_name = "ag_news"  # Change this to the dataset you want to use, e.g., "ag_news", "imdb", etc.
sae, cfg_dict, sparsity = SAE.from_pretrained(
                        release = release,
                        sae_id = sae_id,
                        device = device
)
sae_hook_name_post = f"{sae.cfg.hook_name}.hook_sae_acts_post"
sae.use_error_term = False # passes the reconstructed activations





if dataset_name == "pos":
    xml_path  = Path("/u/siddique-d1/Moghis/storage/dataset/xl-wsd-data/xl-wsd/training_datasets/semcor_en/semcor_en.data.xml")
    out_dir   = Path("/u/siddique-d1/Moghis/storage/dataset/xl-wsd-data/xl-wsd/training_datasets/semcor_en")
    out_csv   = out_dir / "semcor_en_pos_triplets_alltokens.csv"
    out_train = out_dir / "semcor_en_pos_triplets_train.csv"
    out_test  = out_dir / "semcor_en_pos_triplets_test.csv"

    if out_train.exists() and out_test.exists():
        train_results = pd.read_csv(out_train)
        test_results  = pd.read_csv(out_test)
    elif out_csv.exists():
        results = pd.read_csv(out_csv)
        train_results, test_results = split_stratified(results, seed=seed)
        train_results.to_csv(out_train, index=False)
        test_results.to_csv(out_test, index=False)
    else:
        records = build_triplets(xml_path)
        results = pd.DataFrame(records)
        results.to_csv(out_csv, index=False)

        train_results, test_results = split_stratified(results, seed=seed)
        train_results.to_csv(out_train, index=False)
        test_results.to_csv(out_test, index=False)




    train_texts = train_results["text"]
    train_words = train_results["word"]
    train_words = train_words + " is"
    #if model == "gemma-2-2b":
    #    train_words += "<strong>"
    train_pos = train_results["pos"]
    mapping = {
        "NOUN": " noun",
        "ADV": " adverb",
        "ADJ": " adjective",
        "VERB": " verb"
    }
    # Apply mapping
    train_pos = train_pos.map(mapping)


    train_counts = 100_000
    train_texts_final, train_words_final, train_label_final = zip(*random.sample(list(zip(train_texts, train_words, train_pos)), train_counts))

    context_length = 512
    batch_size = 8



    in_context_learning_prompt = "Identify the part of speech (POS) category of the requested word. The only valid categories are: noun, verb, adverb, adjective.\n\n" + \
                                f"Text: By dealing with common landscape in an uncommon way, Roy Mason has found a particular niche in American art.\nThe part of speech category for the word landscape is noun.\n\n" + \
                                f"Text: Their writing, born of their experiments in marijuana and untrammeled sexuality, reflects the extremity of their existential alienation.\nThe part of speech category for the word reflects is verb.\n\n" + \
                                f"Text: The local law here would hold me till they check clear back home, and maybe more than that.\nThe part of speech category for the word here is adverb.\n\n" + \
                                f"Text: But he further said that it was better politics to let others question the wisdom of administration policies first.\nThe part of speech category for the word politics is adjective.\n\n" + \
                                f"Text: "


    end_of_text = "\nThe part of speech category for the word "
elif dataset_name == "ner":    
    # --- load Few-NERD (supervised) ---
    ds = load_dataset("DFKI-SLT/few-nerd", "supervised")
    train = ds["train"]
    train_dict = convert_few_nerd_to_dict(train)
    
    train_texts = train_dict["text"]
    train_words = train_dict["word"]
    train_words = train_words + " is"
    train_ner = train_dict["ner"]
    
    mapping = {
        "person": " person",
        "art": " art",
        "organization": " organization",
        "product": " product",
        "event": " event",
        "location": " location",
        "building": " building"
    }
    # Apply mapping
    train_ner = train_ner.map(mapping)
    
    train_counts = 100_000
    train_texts_final, train_words_final, train_label_final = zip(*random.sample(list(zip(train_texts, train_words, train_ner)), train_counts))
    
    context_length = 512
    batch_size = 4
    
    in_context_learning_prompt = "Identify the named entity category of the requested word or phrase. The only valid categories are: person, art, organization, product, event, location, building.\n\n" + \
                            f"Text: It hosted races from 1903 to 1914, including a race in 1905 AAA Championship Car season won by Louis Chevrolet.\nThe named entity category for Louis Chevrolet is person.\n\n" + \
                            f"Text: In 1992, another film adaptation of the novel was made, `` L'Atlantide ``, directed by Bob Swaim and starring Tchéky Karyo, Jean Rochefort, Anna Galiena, and the famous Spanish actor, Fernando Rey.\nThe named entity category for L'Atlantide is art.\n" + \
                            f"Text: Pakistani scientists and engineers' working at IAEA became aware of advancing Indian nuclear program towards making the bombs.\nThe named entity category for IAEA is organization.\n\n" + \
                            f"Text: Known locally as `` Fairbottom Bobs `` it is now preserved at the Henry Ford Museum in Dearborn, Michigan.\nThe named entity category for Fairbottom Bobs is product.\n\n" + \
                            f"Text: It was the 86th edition of the Australian Open and was held from 19 January through 1 February 1998.\nThe named entity category for Australian Open is event.\n\n" + \
                            f"Text: He left for Italy in April 1857 and also visited Greece before returning to France in 1861.\nThe named entity category for Greece is location.\n\n" + \
                            f"Text: The Cincinnati Enquirer reported on May 22, 2012 that due to the brawl, the game would be held at U.S. Bank Arena for the next two seasons.\nThe named entity category for U.S. Bank Arena is building.\n\n" + \
                            f"Text: "


    end_of_text = "\nThe named entity category for "

if model == "Llama-3.1-8B":
    tokenizer.pad_token_id = tokenizer.eos_token_id

def train_tokenizer(item):
    try:
        tokens = tokenizer.encode(
            item,
            return_tensors="pt",
        )
    except Exception as e:
        tokens = torch.empty([1, 0], dtype=torch.int32)
    return tokens.to(dtype=torch.int32)


tokenized_prompt_train = tokenize_prompt_word_dataset(
    texts=train_texts_final,
    words=train_words_final,
    tokenize=train_tokenizer,
    pad_token_id=tokenizer.pad_token_id,
    in_context_prompt=in_context_learning_prompt,
    end_of_text=end_of_text,
    context_length=context_length
)




num_samples = tokenized_prompt_train['input_ids'].size(0)





sae_hook_name_post = f"{sae.cfg.hook_name}.hook_sae_acts_post"

logs = []
for start in tqdm(range(0, num_samples, batch_size), desc="Processing batches"):
    end = min(start + batch_size, num_samples)
    
    batch_input_ids = tokenized_prompt_train['input_ids'][start:end].to(device)
    batch_attention_mask = tokenized_prompt_train['attention_mask'][start:end].to(device)
    label_tokens = train_label_final[start:end]
    
    with torch.no_grad():
        logits, cache = finetune_model.run_with_cache_with_saes(batch_input_ids, 
                                                                saes=[sae], 
                                                                names_filter=[sae_hook_name_post], 
                                                                return_type="logits", 
                                                                attention_mask=batch_attention_mask)
        
    sae_feature_activations = cache[sae_hook_name_post]
    if combination_type == "last_token":
        sae_activations = sae_feature_activations[:, -1, :]  
    elif combination_type == "all_tokens_max":
        neg_inf = torch.tensor(-float("inf"), device=sae_feature_activations.device, dtype=sae_feature_activations.dtype)
        mask = batch_attention_mask.unsqueeze(-1).expand_as(sae_feature_activations)
        sae_activations = torch.where(mask.bool(), sae_feature_activations, neg_inf)
        sae_activations, _ = sae_activations.max(dim=1)
        
    data = []
    for i in range(sae_activations.size(0)):
            
        indices = torch.nonzero(sae_activations[i], as_tuple=True)[0]
            
        values = sae_activations[i][indices]
        
        if values.dtype  == torch.bfloat16:
            # zip the indices and values together
            nonzero_pairs = list(zip(indices.cpu().numpy().tolist(), values.cpu().float().numpy().tolist()))
        else:
            # zip the indices and values together
            nonzero_pairs = list(zip(indices.cpu().numpy().tolist(), values.cpu().numpy().tolist()))
            
        # Sort the pairs by value in descending order
        nonzero_pairs.sort(key=lambda x: x[1], reverse=True)
            
        # add it to the list
        data.append(nonzero_pairs)
    
    logits = logits[:, -1, :]  # (batch_size, vocab_size)
    next_tokens = torch.argmax(logits, dim=-1)
    
    predicted_tokens = [
        tokenizer.decode(next_tokens[i].cpu().numpy().tolist())
        for i in range(batch_input_ids.size(0))
    ]
    
    logs.extend(zip(label_tokens, predicted_tokens, data, list(range(start, end))))

if dataset_name == "ner":
    dataset_path = storage_path / "ner" 
elif dataset_name == "pos":
    dataset_path = storage_path / "pos" 
dataset_path.mkdir(parents=True, exist_ok=True)

# save the logs
with open(dataset_path / f"logs_{sae_id.replace('/', '_')}_{combination_type}_{sae_hook_name_post}.json", "w") as f:
    json.dump(logs, f)