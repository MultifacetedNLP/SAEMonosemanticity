from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformer_lens import  HookedTransformer
from tqdm import tqdm
import random
from datasets import load_dataset
import json
from utility import tokenize_prompt_dataset
# import path
from pathlib import Path


device = "cuda"
random.seed(42)
model = "Llama-3.1-8B"  # Change this to the model you want to use (e.g., "gemma-2-2b", "Llama-3.1-8B")
if model == "gemma-2-2b":
    tokenizer =  AutoTokenizer.from_pretrained("google/gemma-2-2b")
    finetune_model = HookedTransformer.from_pretrained("gemma-2-2b", device=device, dtype=torch.float16)
    storage_path = Path("/u/siddique-d1/Moghis/storage")
    hook_name = "blocks.25.hook_resid_post"
elif model == "Llama-3.1-8B":
    tokenizer =  AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    hf_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    finetune_model = HookedTransformer.from_pretrained("meta-llama/Llama-3.1-8B", hf_model=hf_model, device=device, dtype=torch.float16)
    storage_path = Path("/u/siddique-d1/Moghis/storage/Llama")
    hook_name = "blocks.31.hook_resid_post"
    
combination_type = "last_token"  # Options: "last_token", "all_tokens_max"
data_name = "ag_news" # Change this to the dataset you want to use, e.g., "ag_news", "imdb", "dbpedia", etc.

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
    train = data["train"]
    context_length = 512
    batch_size = 8
    train_texts, train_labels = zip(*random.sample(list(zip(train["text"], train["label"])), len(train["text"])))
    
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
    train = data["train"]
    context_length = 512
    batch_size = 8
    train_texts, train_labels = zip(*random.sample(list(zip(train["text"], train["label"])), len(train["text"])))
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
    train = data["train"]
    context_length = 1024
    batch_size = 7
    train_counts = 100_000
    train_texts, train_labels = zip(*random.sample(list(zip(train["content"], train["label"])), train_counts))
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
    data = load_dataset("dair-ai/emotion", "split")
    train = data["train"]
    train_texts, train_labels = zip(*random.sample(list(zip(train["text"], train["label"])), len(train["text"])))
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
    data = load_dataset("stanfordnlp/sst2")
    train = data["train"]
    train_texts, train_labels = zip(*random.sample(list(zip(train["sentence"], train["label"])), len(train["sentence"])))
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

if model == "Llama-3.1-8B":
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenized_prompt_train = tokenize_prompt_dataset(
    texts=train_texts,
    tokenize=train_tokenizer,
    pad_token_id=tokenizer.pad_token_id,
    in_context_prompt=in_context_learning_prompt,
    end_of_text=end_of_text,
    context_length=context_length
)


num_samples = tokenized_prompt_train['input_ids'].size(0)

logs = []

for start in tqdm(range(0, num_samples, batch_size), desc="Processing batches"):
    end = min(start + batch_size, num_samples)
    
    batch_input_ids = tokenized_prompt_train['input_ids'][start:end].to(device)
    batch_attention_mask = tokenized_prompt_train['attention_mask'][start:end].to(device)
    label_tokens = [labels_dict[item] for item in train_labels[start:end]]
    
    with torch.no_grad():
        logits, cache = finetune_model.run_with_cache(batch_input_ids,
                                                    attention_mask=batch_attention_mask,
                                                    names_filter=[hook_name], 
                                                    return_type="logits")
        
    feature_activations = cache[hook_name]
    if combination_type == "last_token":
        activations = feature_activations[:, -1, :].cpu().numpy().tolist()
    elif combination_type == "all_tokens_max":
        neg_inf = torch.tensor(-float("inf"), device=feature_activations.device, dtype=feature_activations.dtype)
        mask = batch_attention_mask.unsqueeze(-1).expand_as(feature_activations)  # still 1 or 0
        feature_activations = torch.where(mask.bool(), feature_activations, neg_inf)
        activations, _ = feature_activations.max(dim=1)
        activations = activations.cpu().numpy().tolist()
    
    logits = logits[:, -1, :]  # (batch_size, vocab_size)
    next_tokens = torch.argmax(logits, dim=-1)
    
    predicted_tokens = [
        tokenizer.decode(next_tokens[i].cpu().numpy().tolist())
        for i in range(batch_input_ids.size(0))
    ]
    
    logs.extend(zip(label_tokens, predicted_tokens, activations, list(range(start, end))))


# create a path for the corresponding dataset
if data_name == "ag_news":
    dataset_path = storage_path / "news" 
elif data_name == "imdb":
    dataset_path = storage_path / "imdb"
elif data_name == "dbpedia":
    dataset_path = storage_path / "dbpedia"
elif data_name == "emotions":
    dataset_path = storage_path / "emotions"
elif data_name == "sst2":
    dataset_path = storage_path / "sst2"
    
dataset_path.mkdir(parents=True, exist_ok=True)

# save the logs
with open(dataset_path / f"logs_{combination_type}_{hook_name}.json", "w") as f:
    json.dump(logs, f)