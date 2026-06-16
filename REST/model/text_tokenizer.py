import torch


def get_text_token(tokenizer, sentence):
    sentence_tokenized = tokenizer.encode(text=sentence, add_special_tokens=True)
    sentence_tokenized = sentence_tokenized[
                         :20]  # if the sentence is longer than 20, then this truncates it to 20 words
    # pad the tokenized sentence
    padded_sent_toks = [0] * 20
    padded_sent_toks[:len(sentence_tokenized)] = sentence_tokenized
    # create a sentence token mask: 1 for real words; 0 for padded tokens
    attention_mask = [0] * 20
    attention_mask[:len(sentence_tokenized)] = [1] * len(sentence_tokenized)
    # convert lists to tensors
    padded_sent_toks = torch.tensor(padded_sent_toks).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)
    return padded_sent_toks,attention_mask