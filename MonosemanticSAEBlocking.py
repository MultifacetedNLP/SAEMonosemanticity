from transformers import AutoTokenizer, AutoModelForCausalLM
import pathlib
import torch
import torch.nn.functional as F
from transformer_lens import  HookedTransformer
import sys
from tqdm import tqdm
import random
import numpy as np
from datasets import load_dataset
from collections import defaultdict
import json
import statistics
from pathlib import Path
from sae_lens import SAE, HookedSAETransformer
from utility import tokenize_prompt_dataset, is_match, convert_few_nerd_to_dict, tokenize_prompt_word_dataset
import pandas as pd
# Insert one level up (so you can import HistogramKDE)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from HistogramKDE import TorchKDEHistogram1DVectorized

random.seed(42)
device = "cuda"
combination_type = "last_token"  # Options: "last_token", "all_tokens_max"
data_name = "ag_news"  # Change this to the dataset you want to use, e.g., "ag_news", "imdb", etc.
model = "gemma-2-2b"  # Change this to the model you want to use (e.g., "gemma-2-2b", "Llama-3.1-8B")
sae_threshold = 0.1
if model == "gemma-2-2b":
    hook_name = "blocks.25.hook_resid_post"
    # sae_id = "layer_25/width_65k/canonical"
    sae_id = "layer_25/width_16k/canonical"
    #sae_id = "layer_25/width_16k/average_l0_285"
    release = "gemma-scope-2b-pt-res-canonical"
    #release = "gemma-scope-2b-pt-res"
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

tokenizer =  AutoTokenizer.from_pretrained(tokenizer_name)

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
    data = load_dataset("fancyzhx/ag_news")
    test = data["test"]
    test_texts, test_labels = zip(*random.sample(list(zip(test["text"], test["label"])), len(test["text"])))
    batch_size = 8
    context_length = 512
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
    
    in_context_learning_prompt = "Classify each news article strictly into one of the following categories: world, sports, business, or science and technology—no other categories are allowed.\n\n" + \
                             f"News: Seven Georgian soldiers wounded as South Ossetia ceasefire violated (AFP) AFP - Sporadic gunfire and shelling took place overnight in the disputed Georgian region of South Ossetia in violation of a fragile ceasefire, wounding seven Georgian servicemen.\nThe topic is world\n\n" + \
                             f"News: US NBA players become the Nightmare Team after epic loss (AFP) AFP - Call them the \"Nightmare Team\".\nThe topic is sports\n\n" + \
                             f"News: Stocks End Sharply Higher as Oil Prices Drop A pull back in oil prices and upbeat outlooks from Wal-Mart and Lowe's prompted new bargain-hunting on Wall Street today.\nThe topic is business\n\n" + \
                             f"News: NASA's Genesis Spacecraft Adjusts Course (AP) AP - NASA's Genesis spacecraft successfully adjusted its course this week as it heads back toward Earth with a sample of solar wind particles, the space agency said Wednesday.\nThe topic is science and technology\n\n" + \
                             f"News: India makes elephants appeal Indian asks Bangladesh to spare the lives of around 100 elephants which have strayed across the border.\nThe topic is world\n\n" + \
                             f"News: League-Leading Galaxy Fires Coach (AP) AP - The Los Angeles Galaxy fired coach Sigi Schmid on Monday despite leading Major League Soccer with 34 points..\nThe topic is sports\n\n" + \
                             f"News: Fleet Bank to lay off workers Hundreds of branch employees at Fleet Bank, one of the largest banks in Philadelphia and now part of Bank of America, are being laid off Wednesday across the Northeast.\nThe topic is business\n\n" + \
                             f"News: Study: Global Warming Could Change Calif. (AP) AP - Global warming could cause dramatically hotter summers and a depleted snow pack in California, leading to a sharp increase in heat-related deaths and jeopardizing the water supply, according to a study released Monday.\nThe topic is science and technology\n\nNews: "
    end_of_text = "\nThe topic is "
    if model == "gemma-2-2b":
        end_of_text += "<strong>"
    
elif data_name == "imdb":
    data = load_dataset("stanfordnlp/imdb")
    test = data["test"]
    batch_size = 8
    context_length = 512
    test_texts, test_labels = zip(*random.sample(list(zip(test["text"], test["label"])), len(test["text"])))
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
    in_context_learning_prompt = "Based on the following examples, classify each text as either positive or negative.\n\n" + \
                             f"Text: I thought this movie was terrible. I\'m Chinese, so I thought everything was totally wrong. Many of the facts were incorrect. The only thing right about Chinese history in the movie was when Wendy\'s mother explained to her husband about the statues that guarded ShiHuangDi. I also thought the fight scenes were very cheesy and fake. Many of the actors and actresses were not very great. Some of the jokes that were supposedly \"funny\" were really stupid. I think this movie should receive the worst possible rating it could get. Disney has really got to get more information about Chinese history if they want to create an extravagant movie. Mulan was quite accurate. Watch this movie if you want to waste some time.\nThe sentiment of the text is negative\n\n" + \
                             f"Text: My family and I have viewed this movie often over the years. It is clean, wholesome, heartbreaking and heartwarming. Showing us the compassion between two families of two countries thousands of miles apart and by the most uncanny of coincidences, it's almost as if the hand of God had to be intervening.<br /><br />5 yo Jodelle Micah Ferland who plays Desi the heart stricken little girl, does a magnificent job of acting her part, and for me she was the Priam choice for the lead role.<br /><br />All in all, a 10 out of 10. There are no downsides to this sweet human story. Children of all ages will tearfully, then joyfully watch this and it will bring the viewing family together with smiles and good feelings.\nThe sentiment of the text is positive\n\nText: "
    end_of_text = "\nThe sentiment of the text is "
    if model == "gemma-2-2b":
        end_of_text += "<strong>"
    
elif data_name == "dbpedia":
    data = load_dataset("fancyzhx/dbpedia_14")
    test = data["test"]
    context_length = 1024
    batch_size = 6
    counts = 25_000
    test_texts, test_labels = zip(*random.sample(list(zip(test["content"], test["label"])), counts))
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
    
    in_context_learning_prompt = "Classify each text based on the examples into one of these categories: company, educational institution, artist, athlete, office holder, means of transportation, building, natural place, village, animal, plant, album, film, or written work.\n\n" + \
                             f"Text: Target Canada Co. is the Canadian subsidiary of United States-based discount department store chain Target Corporation formed in 2011 to oversee the company's Canadian operations.\nThe class of the text is company.\n\n" + \
                             f"Text: The Archbishop Lanfranc School is a comprehensive secondary school in the Thornton Heath area of Croydon South London named after Lanfranc Archbishop of Canterbury from 1070 to 1089.\nThe class of the text is educational institution.\n\n" + \
                             f"Text: Emanuel Lee Lambert Jr. (born December 15 1977 in Philadelphia Pennsylvania) is a Christian rapper who goes by his stage name Da' T.R.U.T.H.. \nThe class of the text is artist.\n\n" + \
                             f"Text: Elena Yakovishina (born September 17 1992 in Petropavlovsk-Kamchatsky Russia) is an alpine skier from Russia. She competed for Russia at the 2014 Winter Olympics in the alpine skiing events.\nThe class of the text is athlete.\n\n" + \
                             f"Text: Augustus S. Porter (January 18 1769 – June 10 1849) was an American businessman judge farmer and politician who served as an Assemblyman for the state of New York.\nThe class of the text is office holder.\n\n" + \
                             f"Text: The Yamaha V-Max called the VMAX since 2008 is a cruiser motorcycle made by Yamaha since 1985 known for its powerful V4 engine shaft drive and distinctive styling.\nThe class of the text is means of transportation.\n\n" + \
                             f"Text: The Islamic Museum of Tripoli is a museum of Islamic culture that has been being built under the support and patronage of Saif al-Islam Gaddafi in Tripoli Libya.\n The class of the text is building.\n\n" + \
                             f"Text: The Duruitoarea River is a tributary of the Camenca River in Romania.\nThe class of the text is natural place.\n\n" + \
                             f"Text: Chayly is a village in the Qabala Rayon of Azerbaijan.It is suspected that this village has undergone a name change or no longer exists as no Azerbaijani website mentions it under this name.\nThe class of the text is village.\n\n" + \
                             f"Text: Halaiba is a genus of fly in the family Dolichopodidae.\nThe class of the text is animal.\n\n" + \
                             f"Text: Nectandra matogrossensis is a species of plant in the Lauraceae family. It is endemic to Brazil.\nThe class of the text is plant.\n\n" + \
                             f"Text: Da Drought 3 is a two-disc mixtape by Lil Wayne released on April 13 2007.\nThe class of the text is album.\n\n" + \
                             f"Text: The Nuisance is a 1933 film starring Lee Tracy as a lawyer Madge Evans as his love interest (with a secret) and Frank Morgan as his accomplice.\nThe class of the text is film.\n\n" + \
                             f"Text: Everybody Loves a Good Drought is a book written by P. Sainath about his research findings of poverty in the rural districts of India. The book won him the Magsaysay Award.\nThe class of the text is written work.\n\nText: " 

    end_of_text = "\nThe class of the text is "
    if model == "gemma-2-2b":
        end_of_text += "<strong>"
        
elif data_name == "emotions":
    log_path = storage_path / "emotions"
    data = load_dataset("dair-ai/emotion", "split")
    test = data["test"]
    test_texts, test_labels = zip(*random.sample(list(zip(test["text"], test["label"])), len(test["text"])))
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
    context_length = 512
    batch_size = 8
    in_context_learning_prompt = "Classify the emotion expressed in each text into one of these emotions: sadness, joy, love, anger, fear, or surprise.\n\n" + \
                                 f"Text: i feel like a miserable piece of garbage\nThe emotion of the text is sadness.\n\n" + \
                                 f"Text: i feel very happy and excited since i learned so many things\nThe emotion of the text is joy.\n\n" + \
                                 f"Text: i feel romantic and passionate toward my partner\nThe emotion of the text is love.\n\n" + \
                                 f"Text: i just feel really violent right now\nThe emotion of the text is anger.\n\n" + \
                                 f"Text: i think im just being stupid feeling nervous\nThe emotion of the text is fear.\n\n" + \
                                 f"Text: i feel surprised because i didnt expect it\nThe emotion of the text is surprise.\n\nText: "
    end_of_text = "\nThe emotion of the text is "
    if model == "gemma-2-2b":
        end_of_text += "<strong>"

elif data_name == "sst2":
    log_path = storage_path / "sst2"
    data = load_dataset("stanfordnlp/sst2")
    test = data["validation"]
    test_texts, test_labels = zip(*random.sample(list(zip(test["sentence"], test["label"])), len(test["sentence"])))
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
    context_length = 512
    batch_size = 8
    in_context_learning_prompt = "Based on the following examples, classify the sentiment of each text as either positive or negative.\n\n" + \
                                 f"Text: goes to absurd lengths\nThe sentiment of the text is negative.\n\n" + \
                                 f"Text: the greatest musicians\nThe sentiment of the text is positive.\n\nText: "
    end_of_text = "\nThe sentiment of the text is "
    if model == "gemma-2-2b":
        end_of_text += "<strong>"
        
elif data_name == "pos":
    log_path = storage_path / "pos"
    out_dir   = Path("/u/siddique-d1/Moghis/storage/dataset/xl-wsd-data/xl-wsd/training_datasets/semcor_en")
    out_test  = out_dir / "semcor_en_pos_triplets_test.csv"
    test_results  = pd.read_csv(out_test)
    
    test_texts = test_results["text"]
    test_words = test_results["word"]
    test_words = test_words + " is"
    #if model == "gemma-2-2b":
    #    test_words += "<strong>"
    test_pos = test_results["pos"]
    mapping = {
        "NOUN": " noun",
        "ADV": " adverb",
        "ADJ": " adjective",
        "VERB": " verb"
    }
    # Apply mapping
    test_pos = test_pos.map(mapping)


    test_counts = 10_000
    test_texts_final, test_words_final, test_label_final = zip(*random.sample(list(zip(test_texts, test_words, test_pos)), test_counts))

    context_length = 512
    batch_size = 8
    in_context_learning_prompt = "Identify the part of speech (POS) category of the requested word. The only valid categories are: noun, verb, adverb, adjective.\n\n" + \
                             f"Text: By dealing with common landscape in an uncommon way, Roy Mason has found a particular niche in American art.\nThe part of speech category for the word landscape is noun.\n\n" + \
                             f"Text: Their writing, born of their experiments in marijuana and untrammeled sexuality, reflects the extremity of their existential alienation.\nThe part of speech category for the word reflects is verb.\n\n" + \
                             f"Text: The local law here would hold me till they check clear back home, and maybe more than that.\nThe part of speech category for the word here is adverb.\n\n" + \
                             f"Text: But he further said that it was better politics to let others question the wisdom of administration policies first.\nThe part of speech category for the word politics is adjective.\n\n" + \
                             f"Text: "
    end_of_text = "\nThe part of speech category for the word "
    
elif data_name == "ner":
    log_path = storage_path / "ner"
    ds = load_dataset("DFKI-SLT/few-nerd", "supervised")
    test = ds["test"]
    test_dict = convert_few_nerd_to_dict(test)
    test_texts = test_dict["text"]
    test_words = test_dict["word"]
    test_words = test_words + " is"
    test_ner = test_dict["ner"]
    
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
    test_ner = test_ner.map(mapping)
    
    test_counts = 10_000
    test_texts_final, test_words_final, test_label_final = zip(*random.sample(list(zip(test_texts, test_words, test_ner)), test_counts))
    
    context_length = 512
    batch_size = 8
    
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


if data_name == "pos" or data_name == "ner":
    classes = list(mapping.values())
else:
    classes = list(labels_dict.values())

if model == "Llama-3.1-8B":
    tokenizer.pad_token_id = tokenizer.eos_token_id

if data_name == "pos" or data_name == "ner":
    tokenized_prompt_test = tokenize_prompt_word_dataset(
    texts=test_texts_final,
    words=test_words_final,
    tokenize=train_tokenizer,
    pad_token_id=tokenizer.pad_token_id,
    in_context_prompt=in_context_learning_prompt,
    end_of_text=end_of_text,
    context_length=context_length
    )
else:
    tokenized_prompt_test = tokenize_prompt_dataset(
        texts=test_texts,
        tokenize=train_tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        in_context_prompt=in_context_learning_prompt,
        end_of_text=end_of_text,
        context_length=context_length
    )

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
        for feature, activation_value in activated_features:
            all_freqs[class_index][feature].append(activation_value)


for i, frequencies in enumerate(all_freqs):
    all_freqs[i] = {key: value for key, value in frequencies.items() if len(value) >= sae_threshold * all_counts[i]}


if data_name == "pos" or data_name == "ner":
    possible_labels_ids = []
    for class_name in classes:
        possible_labels_ids.append(tokenizer.encode(class_name)[1])
else:
    if model == "gemma-2-2b":
        # flexible possible_labels_ids
        possible_labels_ids = []
        for class_name in classes:
            possible_labels_ids.append(tokenizer.convert_tokens_to_ids(class_name))
    else:
        possible_labels_ids = []
        for class_name in classes:
            possible_labels_ids.append(tokenizer.encode(class_name)[1])



main_class_drop = 0
other_class_drop = 0
main_class_confidence_drop = 0
other_class_confidence_drop = 0
main_class_ranking_drop = 0
other_class_ranking_drop = 0
for class_neuron in classes:
    
    print(f"{class_neuron} is being blocked:")
    
    frequencies = all_freqs[classes.index(class_neuron)]
    other_frequencies = [all_freqs[i] for i in range(len(classes)) if classes[i] != class_neuron and all_freqs[i]]
    
    # skip to the next cals if empty
    if not frequencies:
        print(f"No frequencies for class {class_neuron}, skipping...")
        continue
    
    keys = list(frequencies.keys())
    other_keys = []
    for other_freq in other_frequencies:
        other_keys.extend(other_freq.keys())
        
    monosemantic_neurons = [neuron for neuron in keys if neuron not in other_keys]


    num_samples = tokenized_prompt_test['input_ids'].size(0)

    predictions = []
    for start in tqdm(range(0, num_samples, batch_size), desc="Processing batches"):
        end = min(start + batch_size, num_samples)
        
        batch_input_ids = tokenized_prompt_test['input_ids'][start:end].to(device)
        batch_attention_mask = tokenized_prompt_test['attention_mask'][start:end].to(device)
        
        if data_name == "pos" or data_name == "ner":
            label_tokens = test_label_final[start:end]
        else:
            label_tokens = [labels_dict[item] for item in test_labels[start:end]]
        
        
        with torch.no_grad():
            logits = finetune_model.run_with_saes(batch_input_ids, 
                                                saes=[sae],
                                                return_type="logits", 
                                                attention_mask=batch_attention_mask)
            
        logits = logits[:, -1, :]
        # apply softmax to the logits
        probs = F.softmax(logits, dim=-1)
        tokens = torch.argmax(probs, dim=-1)
        predicted_tokens_before_blocking = [
            tokenizer.decode(tokens[i].cpu().numpy().tolist())
            for i in range(batch_input_ids.size(0))
        ]
        tokens_probs = probs[:, possible_labels_ids]
        # get the ranking of each possible_labels_ids
        rankings = (probs.unsqueeze(2) > tokens_probs.unsqueeze(1)).sum(dim=1).cpu().tolist()
        tokens_probs = tokens_probs.cpu().tolist()
        
        
        def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
            
            activation[:, -1, monosemantic_neurons] = 0
                
            return activation
        
        with torch.no_grad():
            blocked_logits = finetune_model.run_with_hooks_with_saes(batch_input_ids, 
                                                                    saes=[sae],
                                                                    return_type="logits", 
                                                                    attention_mask=batch_attention_mask,
                                                                    fwd_hooks=[
                                                                            (sae_hook_name_post, hook_fn),
                                                                    ])
                
        
        blocked_logits = blocked_logits[:, -1, :]
        # apply softmax to the logits
        blocked_probs = F.softmax(blocked_logits, dim=-1)
        blocked_tokens = torch.argmax(blocked_probs, dim=-1)
            
        predicted_tokens_after_blocking = [
            tokenizer.decode(blocked_tokens[i].cpu().numpy().tolist())
            for i in range(batch_input_ids.size(0))
        ]
        tokens_probs_after_blocking = blocked_probs[:, possible_labels_ids]
        # get the ranking of each possible_labels_ids
        rankings_after_blocking = (blocked_probs.unsqueeze(2) > tokens_probs_after_blocking.unsqueeze(1)).sum(dim=1).cpu().tolist()
        tokens_probs_after_blocking = tokens_probs_after_blocking.cpu().tolist()
        
        
        predictions.extend(zip(label_tokens, predicted_tokens_after_blocking, predicted_tokens_before_blocking, tokens_probs, tokens_probs_after_blocking, rankings, rankings_after_blocking))


    
    all_smaples_count = [0 for _ in range(len(classes))]
    all_smaples_count_after_blocking = [0 for _ in range(len(classes))]
    all_smaples_count_before_blocking = [0 for _ in range(len(classes))]
    all_confidence_drop = [[] for _ in range(len(classes))]
    all_ranking_drop = [[] for _ in range(len(classes))]

    for label, predicted_after_blocking, predicted_before_blocking, token_probs, token_probs_after_blocking, rankings, rankings_after_blocking in predictions:
        
        for i, class_name in enumerate(classes):
            if label == class_name:
                all_smaples_count[i] = all_smaples_count[i] + 1
                if predicted_before_blocking == class_name:
                    all_confidence_drop[i].append(token_probs[i] - token_probs_after_blocking[i])
                    all_ranking_drop[i].append(rankings_after_blocking[i] - rankings[i])
                    all_smaples_count_before_blocking[i] = all_smaples_count_before_blocking[i] + 1
                    if is_match(predicted_after_blocking, class_name):
                        all_smaples_count_after_blocking[i] = all_smaples_count_after_blocking[i] + 1
    
    accuracy_drops = []
    confidence_drops = []
    ranking_drops = []
    for i, class_name in enumerate(classes):
        accuracy_drop = (all_smaples_count_before_blocking[i] / all_smaples_count[i]) - (all_smaples_count_after_blocking[i] / all_smaples_count[i])
        accuracy_drops.append(accuracy_drop)
        confidence_drop_avg = statistics.mean(all_confidence_drop[i]) if all_confidence_drop[i] else 0
        ranking_drop_avg = statistics.mean(all_ranking_drop[i]) if all_ranking_drop[i] else 0
        confidence_drops.append(confidence_drop_avg)
        ranking_drops.append(ranking_drop_avg)
    
    index = classes.index(class_neuron)
    main_class_drop += accuracy_drops[index]
    main_class_confidence_drop += confidence_drops[index]
    main_class_ranking_drop += ranking_drops[index]
    other_accuracy_drop = sum(accuracy_drops) - accuracy_drops[index]
    other_class_drop += other_accuracy_drop
    other_accuracy_drop_avg = other_accuracy_drop / (len(classes) - 1)
    other_confidence_drop = sum(confidence_drops) - confidence_drops[index]
    other_class_confidence_drop += other_confidence_drop
    other_confidence_drop_avg = other_confidence_drop / (len(classes) - 1)
    other_ranking_drop = sum(ranking_drops) - ranking_drops[index]
    other_class_ranking_drop += other_ranking_drop
    other_ranking_drop_avg = other_ranking_drop / (len(classes) - 1)
    print(f"Class: {class_neuron}")
    print(f"Accuracy drop: {accuracy_drops[index]}")
    print(f"Confidence drop average: {confidence_drops[index]}")
    print(f"Ranking drop average: {ranking_drops[index]}")
    print(f"Other classes accuracy drop average: {other_accuracy_drop_avg}")
    print(f"Other classes confidence drop average: {other_confidence_drop_avg}")
    print(f"Other classes ranking drop average: {other_ranking_drop_avg}")

    print("####################################################################")

print(f"Final Results:")
print(f"Average drop in accuracy of the main class: {main_class_drop/len(classes)}")
print(f"Average drop in accuracy of the other classes: {other_class_drop/( (len(classes) - 1) * len(classes) ) }")
print(f"Average drop in confidence of the main class: {main_class_confidence_drop/len(classes)}")
print(f"Average drop in confidence of the other classes: {other_class_confidence_drop/((len(classes) - 1) * len(classes))}")
print(f"Average drop in ranking of the main class: {main_class_ranking_drop/len(classes)}")
print(f"Average drop in ranking of the other classes: {other_class_ranking_drop/((len(classes) - 1) * len(classes))}")