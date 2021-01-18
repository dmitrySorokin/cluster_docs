import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
import sqlite3
import pandas as pd
import tqdm
import nltk


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()


def bert(text):
    # Load pre-trained model tokenizer (vocabulary)
    marked_text = "[CLS] " + text + " [SEP]"

    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    batch_i = 0
    # Convert the hidden state embeddings into single token vectors

    # Holds the list of 12 layer embeddings for each token
    # Will have the shape: [# tokens, # layers, # features]
    token_embeddings = []

    # For each token in the sentence...
    for token_i in range(len(tokenized_text)):
        # Holds 12 layers of hidden states for each token
        hidden_layers = []

        # For each of the 12 layers...
        for layer_i in range(len(encoded_layers)):
            # Lookup the vector for `token_i` in `layer_i`
            vec = encoded_layers[layer_i][batch_i][token_i]
            hidden_layers.append(vec)
        token_embeddings.append(hidden_layers)

    # [number_of_tokens, 768]
    summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0).numpy() for layer in token_embeddings]
    return np.mean(summed_last_4_layers, axis=0)


if __name__ == '__main__':
    conn = sqlite3.connect(f'parsers/20ng.sqlite')
    raw_files = pd.read_sql('SELECT * FROM RawFiles', conn)

    texts = raw_files['text']
    text_ids = list(map(int, raw_files['file_id']))

    cursor = conn.cursor()
    cursor.execute(f'DROP TABLE IF EXISTS bert')
    cursor.execute(f'CREATE TABLE bert(file_id INTEGER NOT NULL PRIMARY KEY, vec TEXT NOT NULL)')
    for text_id, text in tqdm.tqdm(zip(text_ids, texts)):
        vecs = []
        for sent_text in nltk.sent_tokenize(text):
            vecs.append(bert(sent_text))
        v = np.mean(vecs, axis=0)
        print(len(v))
        vec = ','.join(map(str, v))
        cursor.execute(f'INSERT INTO bert VALUES ({text_id}, "{vec}")')
    conn.commit()
