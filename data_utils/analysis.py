import pickle
from collections import Counter
from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
stopwords = set(stopwords.words("english"))

def analysis_text(text_arr):
    all_tokens = []
    for text in tqdm(text_arr):
        tokens = word_tokenize(text)
        all_tokens.extend(tokens)

    print(Counter(all_tokens))

def analysis_webis(root_dir):
    data_dir = '{}/preprocessed_ne/webis.pickle'.format(root_dir)
    with open(data_dir, 'rb') as handle:
        # ['id', 'portal', 'political_orientation', 'veracity_label', 'text', 'author', 'title', 'ne_text']
        data = pickle.load(handle)

        show_political = False
        if show_political:
            true = [d['political_orientation'] for d in data if d['veracity_label'] == 'mostly true']
            false = [d['political_orientation'] for d in data if d['veracity_label'] != 'mostly true']
            print("True", Counter(true))
            print("False", Counter(false))

        real = [d['text'] for d in data if d['veracity_label'] == 'mostly true']
        fake = [d['text'] for d in data if d['veracity_label'] != 'mostly true']

        print("Real total ", len(real))
        print("Fake total ", len(fake))

        real_tokens, fake_tokens = [], []
        for text in real:
            real_tokens.extend(list(set([w for w in word_tokenize(text.lower()) if w not in stopwords])))

        for text in fake:
            fake_tokens.extend(list(set([w for w in word_tokenize(text.lower()) if w not in stopwords])))

        real_counter = Counter(real_tokens)
        fake_counter = Counter(fake_tokens)

        print("==========REAL==========")
        print("Trump: ", real_counter['trump'], int(real_counter['trump'])/len(real))
        print("Obama: ", real_counter['obama'], int(real_counter['obama'])/len(real))
        print("hillary: ", real_counter['hillary'], int(real_counter['hillary'])/len(real))

        print("\n==========FAKE==========")
        print("Trump: ", fake_counter['trump'], int(fake_counter['trump'])/len(fake))
        print("Obama: ", fake_counter['obama'], int(fake_counter['obama'])/len(fake))
        print("hillary: ", fake_counter['hillary'], int(fake_counter['hillary'])/len(fake))

        print("\n\n\n==========REAL==========")
        print(real_counter.most_common(100))

        print("\n\n\n==========FAKE==========")
        print(fake_counter.most_common(100))

def analysis_clickbait(root_dir):
    data_dir = '{}/preprocessed_ne/clickbait_sns.pickle'.format(root_dir)
    with open(data_dir, 'rb') as handle:
        data = pickle.load(handle)

        real = [d['postText'] for d in data if d['label'] == 'no-clickbait']
        fake = [d['postText'] for d in data if d['label'] == 'clickbait']

        print("Real total ", len(real))
        print("Fake total ", len(fake))

        real_tokens, fake_tokens = [], []
        for text in real:
            real_tokens.extend(list(set([w for w in word_tokenize(text.lower()) if w not in stopwords])))

        for text in fake:
            fake_tokens.extend(list(set([w for w in word_tokenize(text.lower()) if w not in stopwords])))

        real_counter = Counter(real_tokens)
        fake_counter = Counter(fake_tokens)

        print("==========REAL==========")
        print("Trump: ", real_counter['trump'], int(real_counter['trump']) / len(real))
        print("Obama: ", real_counter['obama'], int(real_counter['obama']) / len(real))
        print("hillary: ", real_counter['hillary'], int(real_counter['hillary']) / len(real))

        print("\n==========FAKE==========")
        print("Trump: ", fake_counter['trump'], int(fake_counter['trump']) / len(fake))
        print("Obama: ", fake_counter['obama'], int(fake_counter['obama']) / len(fake))
        print("hillary: ", fake_counter['hillary'], int(fake_counter['hillary']) / len(fake))

        print("\n\n\n==========REAL==========")
        print(real_counter.most_common(100))

        print("\n\n\n==========FAKE==========")
        print(fake_counter.most_common(100))


def analysis_basil(root_dir):
    data_dir = '{}/preprocessed_ne/basil/bias_existence_basil.pickle'.format(root_dir)
    with open(data_dir, 'rb') as handle:
        # ['sentence', 'sentence-index', 'annotations', 'main-event', 'title', 'article_id', 'ne_text', 'label']
        data = pickle.load(handle)
        new_data = []
        for d in data:
            d['orientation'] = d['article_id'].split("_")[1].split(".")[0].lower()
            new_data.append(d)

        show_political = False
        if show_political:
            real = [d['orientation'] for d in data if d['label'] == 'no-bias']
            fake = [d['orientation'] for d in data if d['label'] == 'contain-bias']
            print("real", Counter(real))
            print("fake", Counter(fake))

        real = [d['sentence'] for d in data if d['label'] == 'no-bias']
        fake = [d['sentence'] for d in data if d['label'] == 'contain-bias']

        print("Real total ", len(real))
        print("Fake total ", len(fake))

        real_tokens, fake_tokens = [], []
        for text in real:
            real_tokens.extend(list(set([w for w in word_tokenize(text.lower()) if w not in stopwords])))

        for text in fake:
            fake_tokens.extend(list(set([w for w in word_tokenize(text.lower()) if w not in stopwords])))

        real_counter = Counter(real_tokens)
        fake_counter = Counter(fake_tokens)

        print("==========REAL==========")
        print("Trump: ", real_counter['trump'], int(real_counter['trump'])/len(real))
        print("Obama: ", real_counter['obama'], int(real_counter['obama'])/len(real))
        print("hillary: ", real_counter['hillary'], int(real_counter['hillary'])/len(real))

        print("\n==========FAKE==========")
        print("Trump: ", fake_counter['trump'], int(fake_counter['trump'])/len(fake))
        print("Obama: ", fake_counter['obama'], int(fake_counter['obama'])/len(fake))
        print("hillary: ", fake_counter['hillary'], int(fake_counter['hillary'])/len(fake))

        print("\n\n\n==========REAL==========")
        print(real_counter.most_common(100))

        print("\n\n\n==========FAKE==========")
        print(fake_counter.most_common(100))


if __name__ == "__main__":
    # root_path = "/private/home/nayeon7lee/misinfo_data/"
    # analysis_webis(root_path)
    # analysis_basil(root_path)
    # analysis_clickbait(root_path)

    error_log_path = "/private/home/nayeon7lee/misinfo/error_analysis"

    for f in [f for f in listdir(error_log_path) if isfile(join(error_log_path, f))]:
        print("==========={}===========".format(f))
        file_path = join(error_log_path, f)
        df = pd.read_csv(file_path)
        text_arr = df['texts'].tolist()
        analysis_text(text_arr)
        print("\n\n")