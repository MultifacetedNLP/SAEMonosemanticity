from transformers import AutoTokenizer, AutoModelForCausalLM
import pathlib
import torch
import torch.nn.functional as F
from transformer_lens import  HookedTransformer
import sys
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from collections import defaultdict
import json
from pathlib import Path
from sae_lens import SAE, HookedSAETransformer
from utility import tokenize_analyzing_dataset
import math
# Insert one level up (so you can import HistogramKDE)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from HistogramKDE import TorchKDEHistogram1DVectorized


device = "cuda"
combination_type = "last_token"  # Options: "last_token", "all_tokens_max"
data_name = "ner"  # Change this to the dataset you want to use, e.g., "ag_news", "imdb", etc.
use_sae = True  # Set to True if you want to use SAE
model = "gemma-2-2b"  # Change this to the model you want to use (e.g., "gemma-2-2b", "Llama-3.1-8B")
sae_threshold = 0.1
if model == "gemma-2-2b":
    hook_name = "blocks.25.hook_resid_post"
    #sae_id = "layer_25/width_65k/canonical"
    sae_id = "layer_25/width_16k/canonical"
    release = "gemma-scope-2b-pt-res-canonical"
    storage_path = Path("/u/siddique-d1/Moghis/storage")
    model_name = "gemma-2-2b"
    tokenizer_name = "google/gemma-2-2b"
    hf_model = None
elif model == "Llama-3.1-8B":
    hook_name = "blocks.31.hook_resid_post"
    sae_id = "l31r_800m_slimpajama"
    release = "llama_scope_r1_distill"
    storage_path = Path("/u/siddique-d1/Moghis/storage/Llama")
    model_name = "meta-llama/Llama-3.1-8B"
    tokenizer_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    hf_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    

batch_size = 8

tokenizer =  AutoTokenizer.from_pretrained(tokenizer_name)
if not use_sae:
    finetune_model = HookedTransformer.from_pretrained(model_name, hf_model=hf_model, device=device, dtype=torch.float16)
else:
    finetune_model = HookedSAETransformer.from_pretrained(model_name, hf_model=hf_model, device=device, dtype=torch.float16)
    sae, cfg_dict, sparsity = SAE.from_pretrained(
                            release = release,
                            sae_id = sae_id,
                            device = device
    )
    sae.use_error_term = False # passes the reconstructed activations
    sae_hook_name_post = f"{sae.cfg.hook_name}.hook_sae_acts_post"
        
    
def train_tokenizer(item):
    try:
        tokens = tokenizer.encode(
            item,
            return_tensors="pt",
        )
    except Exception as e:
        tokens = torch.empty([1, 0], dtype=torch.int32)
    return tokens.to(dtype=torch.int32)

if data_name == "ag_news":
    log_path = storage_path / "news"
    if model == "gemma-2-2b":
        labels_dict = {
            0: "world",
            1: "sports",
            2: "business",
            3: "science"
        }
    else:
        labels_dict = {
            0: " world",
            1: " sports",
            2: " business",
            3: " science"
        }
        
elif data_name == "imdb":
    log_path = storage_path / "imdb"
    if model == "gemma-2-2b":
        labels_dict = {
            0: "negative",
            1: "positive"
        }
    else:
        labels_dict = {
            0: " negative",
            1: " positive"
        }
elif data_name == "dbpedia":
    log_path = storage_path / "dbpedia"
    if model == "gemma-2-2b":
        labels_dict = {
            0: "company",
            1: "educational",
            2: "artist",
            3: "athlete",
            4: "office",
            5: "means",
            6: "building",
            7: "natural",
            8: "village",
            9: "animal",
            10: "plant",
            11: "album",
            12: "film",
            13: "written"
        }
    else:
        labels_dict = {
            0: " company",
            1: " educational",
            2: " artist",
            3: " athlete",
            4: " office",
            5: " means",
            6: " building",
            7: " natural",
            8: " village",
            9: " animal",
            10: " plant",
            11: " album",
            12: " film",
            13: " written"
        }
elif data_name == "emotions":
    log_path = storage_path / "emotions"
    if model == "gemma-2-2b":
        labels_dict = {
            0: "sad",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise"
        }
    else:
        labels_dict = {
            0: " sadness",
            1: " joy",
            2: " love",
            3: " anger",
            4: " fear",
            5: " surprise"
        }

elif data_name == "sst2":
    log_path = storage_path / "sst2"
    if model == "gemma-2-2b":
        labels_dict = {
            0: "negative",
            1: "positive"
        }
    else:
        labels_dict = {
            0: " negative",
            1: " positive"
        }
elif data_name == "pos":
    log_path = storage_path / "pos"
    labels_dict = {
        "NOUN": " noun",
        "ADV": " adverb",
        "ADJ": " adjective",
        "VERB": " verb"
    }
elif data_name == "ner":
    log_path = storage_path / "ner"
    labels_dict = {
        "person": " person",
        "art": " art",
        "organization": " organization",
        "product": " product",
        "event": " event",
        "location": " location",
        "building": " building"
    }
    
    
classes = list(labels_dict.values())

# Load 20k sentences from Wikipedia dataset
wiki = load_dataset("wikipedia", "20220301.en", split="train[:20000]", trust_remote_code=True)
# Extract just the text and clean it to sentence-level (simplified)
sentences = [entry['text'] for entry in wiki if entry['text'].strip()]
sentences = sentences[:1000] # TODO: remove this line to use all sentences

if model == "Llama-3.1-8B":
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenized_prompt_test = tokenize_analyzing_dataset(
    texts=sentences,
    tokenize=train_tokenizer,
    pad_token_id=tokenizer.pad_token_id,
    context_length = 512
)

if not use_sae:
    # load the logs
    with open(log_path / f"logs_{combination_type}_{hook_name}.json", "r") as f:
        logs = json.load(f)
else:
    # load the logs
    with open(log_path / f"logs_{sae_id.replace('/', '_')}_{combination_type}_{sae_hook_name_post}.json", "r") as f:
        logs = json.load(f)


# list of defaultdict
all_freqs = []
for cats in classes:
    all_freqs.append(defaultdict(list))
all_counts = [0] * len(classes)

for label, predicted, activated_features, _ in tqdm(logs):
    # if prediction is correct
    if label == predicted:
        class_index = classes.index(label)
        all_counts[class_index] += 1
        if use_sae:
            for feature, activation_value in activated_features:
                all_freqs[class_index][feature].append(activation_value)
        else:
            for feature, activation_value in enumerate(activated_features):
                all_freqs[class_index][feature].append(activation_value)
    
    

if use_sae:
    for i, frequencies in enumerate(all_freqs):
        all_freqs[i] = {key: value for key, value in frequencies.items() if len(value) >= sae_threshold * all_counts[i]}


perplexity_drop = 0
for class_neuron in classes:
    
    print(f"{class_neuron} is being blocked:")
    
    frequencies = all_freqs[classes.index(class_neuron)]
    other_frequencies = [all_freqs[i] for i in range(len(classes)) if classes[i] != class_neuron and all_freqs[i]]
    
    # skip to the next cals if empty
    if not frequencies:
        print(f"No frequencies for class {class_neuron}, skipping...")
        continue
    
    if use_sae:
        keys = list(frequencies.keys())
        other_keys = []
        for other_freq in other_frequencies:
            other_keys.extend(other_freq.keys())
            
        monosemantic_neurons = [neuron for neuron in keys if neuron not in other_keys]
        policemantic_neurons = [neuron for neuron in keys if neuron in other_keys]
    
        
    all_target_data = []
    all_others_data = []
    for other_class in other_frequencies:
        all_others_data.append([])
    
    mu = []
    std = []
    if use_sae:
        process_keys = policemantic_neurons
    else:
        process_keys = frequencies.keys()
    
    for key in tqdm(process_keys, desc="Fitting PDFs", total=len(process_keys)):
        
        data = np.asarray(frequencies[key])
        data = torch.tensor(data, device=device)
        mu.append(data.mean().item())
        std.append(data.std(unbiased=True).item())
        all_target_data.append(data)
        
        for i, other_freq in enumerate(other_frequencies):
            if key in other_freq and len(other_freq[key]) >= 2: 
                data = np.asarray(other_freq[key])
                data = torch.tensor(data, device=device)
                all_others_data[i].append(data)
            else: # for SAE casses happen
                all_others_data[i].append(torch.empty(0, device=device))

    PDF_target = TorchKDEHistogram1DVectorized(bins=2048, bandwidth="scott")
    if use_sae:
        PDF_target.fitSAE(all_target_data)
    else:
        PDF_target.fit(torch.stack(all_target_data, dim=0))
    PDF_others = []
    
    for i, other_data in enumerate(all_others_data):
        PDF_other = TorchKDEHistogram1DVectorized(bins=2048, bandwidth="scott")
        if use_sae:
            PDF_other.fitSAE(other_data)
        else:
            PDF_other.fit(torch.stack(other_data, dim=0))
        PDF_others.append(PDF_other)
    
    all_mus = torch.tensor(mu, device=device)
    all_stds = torch.tensor(std, device=device)
    all_max = all_mus + 2.5 * all_stds
    all_min = all_mus - 2.5 * all_stds


    num_samples = tokenized_prompt_test['input_ids'].size(0)

    total_loss = 0.0
    total_blocked_loss = 0.0
    total_tokens = 0
    for start in tqdm(range(0, num_samples, batch_size), desc="Processing batches"):
        end = min(start + batch_size, num_samples)
        
        batch_input_ids = tokenized_prompt_test['input_ids'][start:end].to(device)
        batch_attention_mask = tokenized_prompt_test['attention_mask'][start:end].to(device)
        
        
        with torch.no_grad():
            if not use_sae:
                loss = finetune_model(batch_input_ids,
                                        attention_mask=batch_attention_mask,
                                        return_type="loss",
                                        loss_per_token=True)
            else:
                loss = finetune_model.run_with_saes(batch_input_ids, 
                                                saes=[sae],
                                                return_type="loss", 
                                                loss_per_token=True,
                                                attention_mask=batch_attention_mask)
            
        next_token_mask = torch.logical_and(batch_attention_mask[:, :-1], batch_attention_mask[:, 1:])
        num_tokens = next_token_mask.sum().item()
        total_tokens += num_tokens
        total_loss += loss.sum().item()
        
        
        def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
            
            if not use_sae:
                activ = activation[:, :, :]
            else:
                activ = activation[:, :, policemantic_neurons]
            
            B, T, D = activ.shape
            activ = activ.view(B * T, D)
                
            all_dampen = []
            for chunk in torch.split(activ, 80, dim=0):
                # chunk: [b, D]
                PDF_target_value = torch.exp(PDF_target.score_samples(chunk.T))
                PDF_others_values = []
                for PDF_other in PDF_others:
                    PDF_other_value = torch.exp(PDF_other.score_samples(chunk.T))
                    PDF_others_values.append(PDF_other_value)
                
                denom = PDF_target_value + sum(PDF_others_values) + 1e-12
                dampen = 1 - (PDF_target_value / denom) 
                
                is_tail = (chunk < all_min) | (chunk > all_max)  # [b, D]
                mask = is_tail.T
                dampen[mask] = 1.0  # set dampen to 1 for tail neurons
                
                all_dampen.append(dampen)
            
            all_dampen = torch.cat(all_dampen, dim=1)  # [D, B*T]
            all_dampen = all_dampen.T  # [B*T, D]
            all_dampen = all_dampen.view(B, T, D)  # [B, T, D]
            
            if not use_sae:
                activation[:, :, :] *= all_dampen  # apply dampening to all neurons
            else:
                activation[:, :, policemantic_neurons] *= all_dampen
                 
            if use_sae:
                activation[:, :, monosemantic_neurons] = 0
                
            return activation
        
        with torch.no_grad():
            if not use_sae:
                blocked_loss = finetune_model.run_with_hooks(batch_input_ids,
                                                                attention_mask=batch_attention_mask,
                                                                return_type="loss",
                                                                loss_per_token=True,
                                                                fwd_hooks=[(hook_name, hook_fn)])
            else:
                blocked_loss = finetune_model.run_with_hooks_with_saes(batch_input_ids, 
                                                                    saes=[sae],
                                                                    return_type="loss", 
                                                                    loss_per_token=True,
                                                                    attention_mask=batch_attention_mask,
                                                                    fwd_hooks=[
                                                                            (sae_hook_name_post, hook_fn),
                                                                    ])
        total_blocked_loss += blocked_loss.sum().item()

    perplexity = math.exp(total_loss / total_tokens)
    perplexity_blocked = math.exp(total_blocked_loss / total_tokens)
    drop = perplexity_blocked - perplexity
    perplexity_drop += drop
    
    print(f"Class: {class_neuron}")
    print(f"Perplexity: {perplexity}")
    print(f"Perplexity (blocked): {perplexity_blocked}")
    print(f"Perplexity drop: {drop}")
    
    print("#" * 20)
    
final_perplexity_drop = perplexity_drop / len(classes)
print(f"Final perplexity drop: {final_perplexity_drop}")