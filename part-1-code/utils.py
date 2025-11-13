import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re

random.seed(0)

NEGATORS = {"not", "n't", "never", "no"}

STRONG_SENT = {
    "good", "great", "amazing", "excellent", "fantastic", "outstanding",
    "bad", "awful", "terrible", "horrible", "worst", "boring", "dull",
    "masterpiece", "mediocre", "love", "hate", "wonderful"
}

STOPWORDS_DROPOUT = {
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "but",
    "if", "when", "while", "as", "at", "by", "it", "this", "that", "there",
    "is", "was", "are", "were", "be", "been", "being"
}

KEYBOARD_NEIGHBORS = {
    "q": ["w"], "w": ["q", "e"], "e": ["w", "r"], "r": ["e", "t"],
    "t": ["r", "y"], "y": ["t", "u"], "u": ["y", "i"], "i": ["u", "o"],
    "o": ["i", "p"], "p": ["o"],
    "a": ["s", "q"], "s": ["a", "d", "w"], "d": ["s", "f", "e"],
    "f": ["d", "g", "r"], "g": ["f", "h", "t"], "h": ["g", "j", "y"],
    "j": ["h", "k", "u"], "k": ["j", "l", "i"], "l": ["k", "o"],
    "z": ["x"], "x": ["z", "c"], "c": ["x", "v"], "v": ["c", "b"],
    "b": ["v", "n"], "n": ["b", "m"], "m": ["n"]
}

def load_data(args, transform_fn=None):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    raw_datasets = load_dataset("imdb")
    if args.debug_train:
        raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42).select(range(100))
        raw_datasets["test"] = raw_datasets["test"].shuffle(seed=42).select(range(100))
    test_valid_split = raw_datasets["test"].train_test_split(test_size=0.1, seed=42)
    raw_datasets["test"] = test_valid_split["train"]
    raw_datasets["validation"] = test_valid_split["test"]
    if transform_fn:
        raw_datasets["test"] = raw_datasets["test"].map(transform_fn)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=args.batch_size)
    test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=args.batch_size)
    return train_dataloader, eval_dataloader, test_dataloader

def example_transform(example):
    example["text"] = example["text"].lower()
    return example

# --- Q2 Transformation Helper Functions ---

def get_wordnet_pos(tag):
    if tag.startswith("J"): return wordnet.ADJ
    if tag.startswith("V"): return wordnet.VERB
    if tag.startswith("N"): return wordnet.NOUN
    if tag.startswith("R"): return wordnet.ADV
    return None

def apply_antonym_negation(tokens, p=0.5, max_replacements=3):
    """
    Replaces words with their antonyms and adds negation.
    e.g., "This movie is good" -> "This movie is not bad"
    """
    if not tokens:
        return tokens

    tagged = pos_tag(tokens)
    new_tokens = list(tokens) # Work on a copy
    replacements = 0
    i = 0
    
    while i < len(tagged) and replacements < max_replacements:
        word, tag = tagged[i]
        lower = word.lower()
        
        if not word.isalpha() or lower in NEGATORS or random.random() > p:
            i += 1
            continue

        wn_pos = get_wordnet_pos(tag)
        if wn_pos not in (wordnet.ADJ, wordnet.ADV): # Only target adj/adv
            i += 1
            continue
        
        antonyms = []
        for syn in wordnet.synsets(lower, pos=wn_pos):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    antonyms.append(lemma.antonyms()[0].name())
        
        if not antonyms:
            i += 1
            continue

        # Found an antonym
        antonym = random.choice(antonyms).replace("_", " ")
        
        # Find the verb to negate (e.g., "is", "was")
        # This is a simple lookbehind
        verb_idx = -1
        if i > 0 and tagged[i-1][0].lower() in {"is", "was", "are", "were"}:
            verb_idx = i - 1
        
        if verb_idx != -1:
            # "is good" -> "is not bad"
            new_tokens[verb_idx] = tagged[verb_idx][0] + " not"
            new_tokens[i] = antonym
            replacements += 1
            i += 1 # Skip the word we just replaced
        elif i > 0 and tagged[i-1][0].lower() == "not":
             # Handle "not good" -> "bad" (removes the negation)
             new_tokens[i-1] = "" # Remove "not"
             new_tokens[i] = antonym
             replacements += 1
             i += 1
        else:
            # Fallback: "good film" -> "not bad film"
            # This is less grammatical but still a valid transformation
            new_tokens[i] = "not " + antonym
            replacements += 1
            i += 1
        
        i += 1
        
    # Filter out any empty strings from "not" removal
    return [t for t in new_tokens if t]


def apply_synonym_replacement(tokens, p=0.6, max_replacements=15): 
    if not tokens: return tokens
    tagged = pos_tag(tokens)
    replacements = 0
    for i, (word, tag) in enumerate(tagged):
        if replacements >= max_replacements: break
        lower = word.lower()
        if not word.isalpha() or lower in NEGATORS or random.random() > p:
            continue
        wn_pos = get_wordnet_pos(tag)
        if wn_pos is None: continue

        synsets = wordnet.synsets(lower, pos=wn_pos)
        candidates = set()
        for syn in synsets:
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != lower and name.isalpha():
                    candidates.add(name)
        if not candidates: continue
        new_word = random.choice(list(candidates))
        if word[0].isupper(): new_word = new_word.capitalize()
        tokens[i] = new_word
        replacements += 1
    return tokens


def apply_typo(tokens, p=0.5, max_typos=10):
    if not tokens: return tokens
    typos = 0
    word_indices = list(range(len(tokens)))
    random.shuffle(word_indices)
    for i in word_indices:
        word = tokens[i]
        if typos >= max_typos: break
        if len(word) < 4 or not any(ch.isalpha() for ch in word): continue
        if word.lower() in NEGATORS or random.random() > p:
            continue
        char_indices = [idx for idx, ch in enumerate(word) if ch.isalpha()]
        if not char_indices: continue
        idx = random.choice(char_indices)
        ch = word[idx].lower()
        if ch not in KEYBOARD_NEIGHBORS: continue
        new_ch = random.choice(KEYBOARD_NEIGHBORS[ch])
        if word[idx].isupper(): new_ch = new_ch.upper()
        tokens[i] = word[:idx] + new_ch + word[idx + 1:]
        typos += 1
    return tokens


def apply_word_dropout(tokens, p=0.4): 
    new_tokens = []
    for w in tokens:
        if w.lower() in STOPWORDS_DROPOUT and random.random() < p:
            continue
        new_tokens.append(w)
    return new_tokens

# --- Active <-> Passive ---
def passive_to_active(sentence):
    if any(neg in sentence.lower().split() for neg in NEGATORS): return sentence
    pattern = re.compile(r"(.+?)\s+was\s+([A-Za-z]+ed)\s+by\s+(.+)", re.IGNORECASE)
    match = pattern.search(sentence)
    if not match: return sentence
    patient, verb, agent = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
    new_sent = f"{agent} {verb} {patient}"
    return pattern.sub(new_sent, sentence, count=1)

def active_to_passive(sentence):
    if any(neg in sentence.lower().split() for neg in NEGATORS): return sentence
    pattern = re.compile(r"(.+?)\s+([A-Za-z]+ed)\s+(me|him|her|us|them)", re.IGNORECASE)
    match = pattern.search(sentence)
    if not match: return sentence
    subj, verb, obj = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
    aux = "were" if obj.lower() in {"us", "them"} else "was"
    new_sent = f"{obj} {aux} {verb} by {subj}"
    return pattern.sub(new_sent, sentence, count=1)

def transform_voice(text, p=0.6): 
    if random.random() > p: return text
    parts = re.split(r"([.!?])", text)
    sent_indices = [i for i in range(0, len(parts), 2) if parts[i].strip()]
    if not sent_indices: return text
    idx = random.choice(sent_indices)
    sent = parts[idx]
    new_sent = passive_to_active(sent)
    if new_sent == sent: new_sent = active_to_passive(sent)
    if new_sent == sent: return text
    parts[idx] = new_sent
    return "".join(parts)


# --- Main Q2 Transformation Function ---

def custom_transform(example):

    text = example["text"]

    # 1) voice transformation
    text = transform_voice(text, p=0.6)

    # 2) Token-level transformations
    tokens = word_tokenize(text)

    # 2a) Antonym Negation 
    tokens = apply_antonym_negation(tokens, p=0.5, max_replacements=3)

    # 2b) Synonym replacement
    tokens = apply_synonym_replacement(tokens, p=0.6, max_replacements=15)

    # 2c) Keyboard typos
    tokens = apply_typo(tokens, p=0.5, max_typos=10)

    # 2d) Stopword dropout
    tokens = apply_word_dropout(tokens, p=0.4)

    # 3) Detokenize
    detok = TreebankWordDetokenizer()
    example["text"] = detok.detokenize(tokens)

    return example