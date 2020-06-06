from tika import parser
import re
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer


_LEM = WordNetLemmatizer()


def normalize(text):
    words = list(filter(lambda x: len(x) > 0, text.split(' ')))
    tagged_tokens = pos_tag(words)#[0][1][0].lower()
    normalized_words = []
    for (word, tag) in tagged_tokens:
        stem_tag = tag[0].lower()
        if stem_tag not in ['a', 's', 'r', 'n', 'v']:
            stem_tag = 'n'
        word = _LEM.lemmatize(word, stem_tag)
        normalized_words.append(word)

    # TODO merge not, merge collocations?
    return ' '.join(normalized_words)


def cleanup(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'([^a-zA-Z ])', '', text)
    return text


def parse(path):
    try:
        content = parser.from_file(path)['content']
    except:
        print('can not parse', path)
        content = None

    if content is None:
        content = ''
        print('can not read', path)

    return content


def process(file_id, item_id, collection_id, collection_name, file_path):
    print('process', file_path)
    text = parse(file_path)
    print('parsed')
    text = cleanup(text)
    print('cleaned')
    text = normalize(text)
    print('normalized')
    return file_id, item_id, collection_id, collection_name, text
