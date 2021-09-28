from os import listdir
from os.path import isfile, join
import json
import pandas as pd
import pickle
from tqdm import tqdm
import spacy
import jsonlines
from bs4 import BeautifulSoup
import random
from collections import defaultdict
import os
import numpy as np
from os import listdir
from os.path import isfile, join
from nltk.tokenize import sent_tokenize

# """
# PERSON	People, including fictional.
# NORP	Nationalities or religious or political groups.
# FAC	Buildings, airports, highways, bridges, etc.
# ORG	Companies, agencies, institutions, etc.
# GPE	Countries, cities, states.
# LOC	Non-GPE locations, mountain ranges, bodies of water.
# PRODUCT	Objects, vehicles, foods, etc. (Not services.)
# EVENT	Named hurricanes, battles, wars, sports events, etc.
# WORK_OF_ART	Titles of books, songs, etc.
# LAW	Named documents made into laws.
# LANGUAGE	Any named language.
# DATE	Absolute or relative dates or periods.
# TIME	Times smaller than a day.
# PERCENT	Percentage, including ”%“.
# MONEY	Monetary values, including unit.
# QUANTITY	Measurements, as of weight or distance.
# ORDINAL	“first”, “second”, etc.
# CARDINAL	Numerals that do not fall under another type.
#
# """

ent2text = {
    "PERSON": "person",
    "NORP": "group",
    "FAC": "facilities",
    "ORG": "organizations",
    "GPE": "country",
    "LOC": "location",
    "PRODUCT": "product",
    "EVENT": "event",
    "WORK_OF_ART": "art",
    "LAW": "law",
    "LANGUAGE": "language",
    "DATE": "date",
    "TIME": "time",
    "PERCENT": "percentage",
    "MONEY": "money",
    "QUANTITY": "quantity",
    "ORDINAL": "number",
    "CARDINAL": "number"
}

def replace_ne_with_special_token(text, nlp):
    doc = nlp(text)
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    for ent in ents:
        text = text.replace(ent[0], ent2text[ent[1]])
    return text

def create_basil_data():
    basil_path = "/private/home/nayeon7lee/misinfo_data/basil/data/"
    save_path = "/private/home/nayeon7lee/misinfo_data/preprocessed_ne/basil"
    f_names = [f_name for f_name in listdir(basil_path) if isfile(join(basil_path, f_name))]

    nlp = spacy.load("en_core_web_sm")

    # read all files and save as Array[ sentence objects ]
    all_sent_objs = []
    for name in tqdm(f_names, total=len(f_names)):
        path = join(basil_path, name)
        with open(path, 'r') as json_file:
            data = json.load(json_file)

            meta_obj = {'main-event': data['main-event'], 'title': data['title'], 'article_id': name}
            for sent_obj in data['body']:
                sent_obj.update(meta_obj)
                sent_obj['ne_text'] = replace_ne_with_special_token(sent_obj['sentence'], nlp)

                all_sent_objs.append(sent_obj)

    # # 1. binary classification: contain_bias vs no_bias
    # bias_existence_basil = []
    # cnt = 0
    # for sent_obj in all_sent_objs:
    #     if sent_obj['annotations'] == []:
    #         sent_obj['label'] = 'no-bias'
    #     else:
    #         sent_obj['label'] = 'contain-bias'
    #     bias_existence_basil.append(sent_obj)
    #
    #
    # with open('{}/bias_existence_basil.pickle'.format(save_path), 'wb') as handle:
    #     pickle.dump(bias_existence_basil, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print("bias_existence_basil", len(bias_existence_basil))

    bias_type_basil = []
    for sent_obj in all_sent_objs:
        # if there exists annotation for bias
        anno_count = len(sent_obj['annotations'])
        if anno_count == 0:
            sent_obj['label'] = 'no-bias'
        elif anno_count == 1:
            sent_obj['label'] = sent_obj['annotations'][0]['bias']
        elif anno_count > 1:
            bias_type = list(set([anno['bias'] for anno in sent_obj['annotations']]))
            if len(bias_type) == 1:
                sent_obj['label'] = bias_type[0]
            else:
                sent_obj['label'] = "both"
        bias_type_basil.append(sent_obj)

    with open('{}/bias_type_basil_label_for_analysis.pickle'.format(save_path), 'wb') as handle:
        pickle.dump(bias_type_basil, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("bias_type_basil", len(bias_type_basil))

    #
    # # 2. bias type classification: political bias vs lexical bias
    # bias_type_basil = []
    # for sent_obj in all_sent_objs:
    #     # if there exists annotation for bias
    #     anno_count = len(sent_obj['annotations'])
    #     if anno_count == 0:
    #         continue
    #         # sent_obj['label'] = 'no-bias'
    #     elif anno_count == 1:
    #         sent_obj['label'] = sent_obj['annotations'][0]['bias']
    #     elif anno_count > 1:
    #         bias_type = list(set([anno['bias'] for anno in sent_obj['annotations']]))
    #         if len(bias_type) == 1:
    #             sent_obj['label'] = bias_type[0]
    #         else:
    #             continue  # sentence containing both lexical and political bias. so skip
    #
    #     bias_type_basil.append(sent_obj)
    # with open('{}/bias_type_basil.pickle'.format(save_path), 'wb') as handle:
    #     pickle.dump(bias_type_basil, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print("bias_type_basil", len(bias_type_basil))
    #
    # # 3. polarity classification
    # polarity_basil = []
    # for sent_obj in all_sent_objs:
    #     anno_count = len(sent_obj['annotations'])
    #     if anno_count == 1:  # only get polarity if there's one bias span. polarity label is for each bias span. cannot use this label for sent level if sent contains more than one span
    #         sent_obj['label'] = sent_obj['annotations'][0]['polarity']
    #     else:
    #         continue
    #     polarity_basil.append(sent_obj)
    # with open('{}/polarity_basil.pickle'.format(save_path), 'wb') as handle:
    #     pickle.dump(polarity_basil, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print("polarity_basil", len(polarity_basil))
    #
    # # 4. seqeunce tagging
    # bias_span_tagging_basil = []
    # for sent_obj in all_sent_objs:
    #     # if there exists annotation for bias
    #     anno_count = len(sent_obj['annotations'])
    #     if anno_count == 0:
    #         continue
    #     else:
    #         for anno in sent_obj['annotations']:
    #             anno.update({'sentence': sent_obj['sentence'],
    #                          'sentence-index': sent_obj['sentence-index'],
    #                          'article_id': sent_obj['article_id']})
    #
    #             bias_span_tagging_basil.append(anno)
    #
    # print(bias_span_tagging_basil[:10])
    # print("bias_span_tagging_basil", len(bias_span_tagging_basil))
    # with open('{}/bias_span_tagging_basil.pickle'.format(save_path), 'wb') as handle:
    #     pickle.dump(bias_span_tagging_basil, handle, protocol=pickle.HIGHEST_PROTOCOL)


def prepare_clickbait(nlp):
    save_path = "/private/home/nayeon7lee/misinfo_data/preprocessed_ne"

    with open('/private/home/nayeon7lee/misinfo_data/clickbait/truth.jsonl', 'r') as label_file:
        label_list = [json.loads(item) for item in list(label_file)]
        id2label_dict = {item['id']: item['truthClass'] for item in label_list}

    with open('/private/home/nayeon7lee/misinfo_data/clickbait/instances.jsonl', 'r') as json_file:
        data_list = [json.loads(item) for item in list(json_file)]

    new_clickbait_data = []
    for data in tqdm(data_list):
        data['label'] = id2label_dict[data['id']]
        data['postText'] = " ".join(data['postText'])
        data['ne_text'] = replace_ne_with_special_token(" ".join(data['postText']), nlp)
        new_clickbait_data.append(data)

    with open('{}/clickbait_sns.pickle'.format(save_path), 'wb') as handle:
        pickle.dump(new_clickbait_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def apply_pos_nei(data, nlp, text_field="text"):
    data_pos = []
    for d in tqdm(data):
        doc = nlp(d[text_field])
        tokens = [(token.text, token.pos_, token.tag_) for token in doc]
        ents = [(ent.text, ent.label_) for ent in doc.ents]

        data_pos.append({text_field: d[text_field], "tokens": tokens, "ents": ents, "label": d['label']})
    return data_pos


def pos_tag_liar_dataset():

    # load data
    with open("/private/home/nayeon7lee/misinfo_data/preprocessed/liar_train.pickle", 'rb') as handle:
        train = pickle.load(handle)
    with open("/private/home/nayeon7lee/misinfo_data/preprocessed/liar_dev.pickle", 'rb') as handle:
        dev = pickle.load(handle)
    with open("/private/home/nayeon7lee/misinfo_data/preprocessed/liar_test.pickle", 'rb') as handle:
        test = pickle.load(handle)

    nlp = spacy.load("en_core_web_sm")

    train_pos = apply_pos_nei(train, nlp)
    dev_pos = apply_pos_nei(dev, nlp)
    test_pos = apply_pos_nei(test, nlp)

    save_path = "/private/home/nayeon7lee/misinfo_data/preprocessed/"
    with jsonlines.open('{}/liar_train_pos.jsonl'.format(save_path), 'w') as writer:
        writer.write_all(train_pos)

    with jsonlines.open('{}/liar_dev_pos.jsonl'.format(save_path), 'w') as writer:
        writer.write_all(dev_pos)

    with jsonlines.open('{}/liar_test_pos.jsonl'.format(save_path), 'w') as writer:
        writer.write_all(test_pos)


def liar_tag_analysis(phase_path='liar_train_pos.jsonl'):
    save_path = "/private/home/nayeon7lee/misinfo_data/preprocessed/"
    useful_claims = []
    exclude_nei = set(['PERCENT', 'CARDINAL', 'MONEY', 'DATE'])
    keep_nei = set(['EVENT', 'PERSON', 'ORG'])
    numerical = 0
    non_num = 0
    with jsonlines.open('{}/{}'.format(save_path, phase_path)) as reader:
        for obj in tqdm(reader):
            ent_types = [e[1] for e in obj['ents']]
            if len(set(ent_types).intersection(exclude_nei)) > 0: # contains numerical stuff. skip
                numerical += 1
            else:
                non_num += 1
                if len(set(ent_types).intersection(keep_nei)) > 0:
                    useful_claims.append(obj)

    # print([(c['text'], c['ents'], c['label']) for c in useful_claims])
    print("claim containing numerical are: ", numerical)
    print("claim containing non-numerical are: ", non_num)
    print("useful claim ", len(useful_claims))


def obtain_nei(text, nlp):
    doc = nlp(text)
    nei_ent = [(ent.text, ent.label_) for ent in doc.ents]
    return nei_ent

def pos_tag_fever(filename):
    nlp = spacy.load("en_core_web_sm")

    objs_with_ne = []
    with jsonlines.open("/private/home/nayeon7lee/misinfo_data/fever/{}.jsonl".format(filename)) as reader:
        for obj in tqdm(reader, total=19998):
            obj['nei'] = obtain_nei(obj['claim'], nlp)
            objs_with_ne.append(obj)

    save_path = "/private/home/nayeon7lee/misinfo_data/fever/{}_ne.jsonl".format(filename)
    with jsonlines.open(save_path, 'w') as writer:
        writer.write_all(objs_with_ne)

def split_fever_by_verifiability(filename, total):
    nlp = spacy.load("en_core_web_sm")

    verifiable, unverifiable = [], []
    with jsonlines.open("/private/home/nayeon7lee/misinfo_data/fever/{}.jsonl".format(filename)) as reader:
        for obj in tqdm(reader, total=total):
            obj['nei'] = obtain_nei(obj['claim'], nlp)
            if obj['verifiable'] == 'VERIFIABLE':
                verifiable.append(obj)
            else:
                unverifiable.append(obj)

    with jsonlines.open("/private/home/nayeon7lee/misinfo_data/fever/{}_ne_verifiable.jsonl".format(filename), 'w') as writer:
        writer.write_all(verifiable)

    with jsonlines.open("/private/home/nayeon7lee/misinfo_data/fever/{}_ne_unverifiable.jsonl".format(filename), 'w') as writer:
        writer.write_all(unverifiable)


def random_sample_fever(filename, total, count):
    all_obj = []
    with jsonlines.open("/private/home/nayeon7lee/misinfo_data/fever/{}_ne_verifiable.jsonl".format(filename)) as reader:
        for obj in tqdm(reader, total=total):
            all_obj.append(obj)

    random.shuffle(all_obj)

    df = pd.DataFrame(columns=['id', 'claim', 'label'])

    for obj in all_obj[:count]:
        df = df.append({'id': obj['id'], 'claim': obj['claim'], 'label': obj['label']}, ignore_index=True)

    save_path = "/private/home/nayeon7lee/misinfo_data/fever/{}_ne_verifiable_random_50_madian.csv".format(filename)
    df.to_csv(save_path, encoding='utf-8', index=False)

def text2question(text, ne, type):
    if type == 'ne':
        print("retrieve knowledge about relevant entity")
    elif type == "brute_force":
        print("try to obtain all relevant information by masking all tokens, but stopwords")
    else:
        print("wrong type given")
        exit(1)

def create_question_fever(filename, total):
    objs_with_q = []
    with jsonlines.open("/private/home/nayeon7lee/misinfo_data/fever/{}_ne_verifiable.jsonl".format(filename)) as reader:
        for obj in tqdm(reader, total=total):
            obj['question'] = text2question(obj['claim'], obj['nei'], type='ne')
            objs_with_q.append(obj)

    with jsonlines.open("/private/home/nayeon7lee/misinfo_data/fever/{}_ne_verifiable_random_50.jsonl".format(filename), 'w') as writer:
        writer.write_all()

def prepare_webis(nlp):
    root_path = "/private/home/nayeon7lee/misinfo_data/"
    annotation_path = os.path.join(root_path, 'webis/overview.csv')
    articles_path = os.path.join(root_path, 'webis/articles')
    save_path = "/private/home/nayeon7lee/misinfo_data/preprocessed_ne"

    # overview: XML,portal,orientation,veracity,url
    annotation_df = pd.read_csv(annotation_path)
    webis_data = []
    for _, anno in tqdm(annotation_df.iterrows(), total=len(annotation_df)):
        # read xml file
        article = {}
        article['id'] = anno['XML']
        article['portal'] = anno['portal']
        article['political_orientation'] = anno['orientation']
        article['veracity_label'] = anno['veracity']
        with open("{}/{}".format(articles_path, anno['XML'])) as fp:
            xml = BeautifulSoup(fp)

            article['text'] = xml.find('maintext').text.encode('utf8').decode('utf8')
            article['author'] = xml.find('author').text
            article['title'] = xml.find('title').text
            article['ne_text'] = replace_ne_with_special_token(article['text'], nlp)

        webis_data += article,

    with open('{}/webis.pickle'.format(save_path), 'wb') as handle:
        pickle.dump(webis_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def prepare_fakenewsnet_politifact():
    root_path = "/home/nayeon/misinfo/data"
    save_path = "/home/nayeon/misinfo/data/preprocessed"
    buzzfeed_dir = '{}/FakeNewsNet/Data/PolitiFact'.format(root_path)

    real_dir = '{}/RealNewsContent'.format(buzzfeed_dir)
    fake_dir = '{}/FakeNewsContent'.format(buzzfeed_dir)

    real_data, fake_data = [], []
    for file_path in [join(real_dir, f) for f in listdir(real_dir) if isfile(join(real_dir, f))]:
        with open(file_path) as json_file:
            obj = json.load(json_file)
            obj['label'] = 'real'
            real_data.append(obj)

    for file_path in [join(fake_dir, f) for f in listdir(fake_dir) if isfile(join(fake_dir, f))]:
        with open(file_path) as json_file:
            obj = json.load(json_file)
            obj['label'] = 'fake'
            fake_data.append(obj)

    all_data = fake_data + real_data
    np.random.shuffle(all_data)

    with open('{}/fnn_politifact.pickle'.format(save_path), 'wb') as handle:
        pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def prepare_fakenewsnet_buzzfeed():
    root_path = "/home/nayeon/misinfo/data"
    save_path = "/home/nayeon/misinfo/data/preprocessed"
    buzzfeed_dir = '{}/FakeNewsNet/Data/BuzzFeed'.format(root_path)

    real_dir = '{}/RealNewsContent'.format(buzzfeed_dir)
    fake_dir = '{}/FakeNewsContent'.format(buzzfeed_dir)

    real_data, fake_data = [], []
    for file_path in [join(real_dir, f) for f in listdir(real_dir) if isfile(join(real_dir, f))]:
        with open(file_path) as json_file:
            obj = json.load(json_file)
            obj['label'] = 'real'
            real_data.append(obj)

    for file_path in [join(fake_dir, f) for f in listdir(fake_dir) if isfile(join(fake_dir, f))]:
        with open(file_path) as json_file:
            obj = json.load(json_file)
            obj['label'] = 'fake'
            fake_data.append(obj)

    all_data = fake_data + real_data
    np.random.shuffle(all_data)

    with open('{}/fnn_buzzfeed.pickle'.format(save_path), 'wb') as handle:
        pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# def prepare_fakenewsnet_gossip():
#     root_path = "/private/home/nayeon7lee/misinfo_data/"
#     save_path = "/private/home/nayeon7lee/misinfo_data/preprocessed"
#     gossip_path = '{}/FakeNewsNet/dataset'.format(root_path)

#     real_df = pd.read_csv("{}/gossipcop_real.csv".format(gossip_path))
#     real_df['label'] = 'real'

#     fake_df = pd.read_csv("{}/gossipcop_fake.csv".format(gossip_path))
#     fake_df['label'] = 'fake'

#     gossip_df = real_df.append(fake_df, ignore_index=True)
#     data = [{"id": row["id"], "url": row["news_url"], "title": row['title'], "label": row["label"]}
#             for index, row in gossip_df.iterrows()]

#     np.random.shuffle(data)


#     with open('{}/fnn_gossip.pickle'.format(save_path), 'wb') as handle:
#         pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def prepare_newstrust():
    folder_path = "/private/home/nayeon7lee/misinfo_data/DeClarE/newstrust/NewsTrustData"
    # NewsArticles.tsv  NewsTrustMembers.tsv	NewsTrustSources.tsv  NewsTrustStories.tsv  README.txt	TopicModel.txt

def prepare_snopes():
    path = "/private/home/nayeon7lee/misinfo_data/DeClarE/Snopes/snopes.tsv"


def propaganda():
    path='/home/nayeon/misinfo/data/propaganda/data/protechn_corpus_eval'

    def read_and_process_files(phase='test', save_path='/home/nayeon/misinfo/data/preprocessed'):
        full_path="{}/{}".format(path, phase)

        txt_onlyfiles = [f for f in listdir(full_path) if ".txt" in f]
        all_data = []

        for idx, txt_path in enumerate(txt_onlyfiles):
            try:
                label_path_ = "{}/{}".format(full_path, txt_path.split('.')[0] + ".labels.tsv")
                labels = pd.read_csv(label_path_, sep='\t')

                with open("{}/{}".format(full_path, txt_path), 'r') as txt_in_f:
                    text = txt_in_f.read()
                
                # to_find_phrases = []
                # for label_row in labels.iterrows():
                #     phrase = text[label_row[1][2]: label_row[1][3]].strip()
                #     to_find_phrases.append(phrase)

                indices = []
                for label_row in labels.iterrows():
                    indices.append((label_row[1][2],label_row[1][3]))

                indices = sorted(indices, key=lambda item: item[0]) 
                # print(indices)
                to_find_phrases = [text[s:e] for (s,e) in indices if len(text[s:e].strip()) > 3]
                # print(to_find_phrases)
                candidates = text.split(".")
                # candidates = sent_tokenize(text)
                candidate_idx, to_find_idx = 0, 0

                found_og_sents = []
                discard_sents = []
                while candidate_idx < len(candidates) and to_find_idx < len(to_find_phrases):
                    if to_find_phrases[to_find_idx] in candidates[candidate_idx]:
                        found_og_sents.append(candidates[candidate_idx])
                        to_find_idx += 1
                    else:
                        discard_sents.append(candidates[candidate_idx])
                    # print(candidates[candidate_idx].strip())
                    # print("\n")
                    candidate_idx+=1
                

                # propaganda samples
                # print(found_og_sents)
                propaganda_samples = [{'text': sent, 'label': 'has_propaganda'} for sent in found_og_sents]
                all_data.extend(propaganda_samples)

                # no-propaganda samples
                random.shuffle(discard_sents)
                negative_samples = discard_sents[:len(found_og_sents)]
                # print(negative_samples)
                non_propaganda_samples = [{'text': sent, 'label': 'no_propaganda'} for sent in negative_samples]
                all_data.extend(non_propaganda_samples)
            except:
                # print("skipped")
                # print(txt_path)
                continue

        
        np.random.shuffle(all_data)
        print(len(all_data))

        with open('{}/propaganda_{}.pickle'.format(save_path, phase), 'wb') as handle:
            pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    read_and_process_files('test')
    read_and_process_files('train')
    read_and_process_files('dev')


def preprocess_twitter_covid():

    twitter_covid_data_path = "/home/nayeon/misinfo/data/twitter_covid19_infodemic_english_data.tsv"
    data_df = pd.read_csv(twitter_covid_data_path, sep='\t')

    q1_df = data_df.loc[:, 'text': 'q1_label']
    q2_df = data_df.loc[:, 'text': 'q2_label']
    q6_df = data_df.loc[:, 'text': 'q6_label']
    q7_df = data_df.loc[:, 'text': 'q7_label']
    # print("before", len(q1_df), len(q2_df), len(q6_df), len(q7_df))

    # filter any NAN label rows
    q1_df=q1_df.dropna()
    q2_df=q2_df.dropna()
    q6_df=q6_df.dropna()
    q7_df=q7_df.dropna()

    print("data sizes:", len(q1_df), len(q2_df), len(q6_df), len(q7_df))
    save_path_template = "/home/nayeon/misinfo/data/preprocessed/{}.pickle"

    for name, label_field_name, df_to_save in zip(["twitter_q1", "twitter_q2", "twitter_q6", "twitter_q7"], ['q1_label','q2_label','q6_label','q7_label'], [q1_df, q2_df, q6_df, q7_df]):
        
        save_path = save_path_template.format(name)
        data_to_save = [{'text': row[1]['text'], 'label': row[1][label_field_name]} for row in df_to_save.iterrows()]
        with open(save_path, 'wb') as handle:
            pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

def process_twitter_covid_q2():
    in_path = "/home/nayeon/misinfo/data/preprocessed/twitter_q2.pickle"

    label_map = {
        "1_no_definitely_contains_no_false_info": "no_false",
        "2_no_probably_contains_no_false_info": "no_false",
        "3_not_sure": "N/A",
        "4_yes_probably_contains_false_info": "contains_false",
        "5_yes_definitely_contains_false_info": "contains_false",

    }

    new_data = []
    with open(in_path, 'rb') as handle:
        data = pickle.load(handle)
        for item in data:
            text_label = item['label']
            if text_label == "3_not_sure":
                continue  # skip
            else:
                label = label_map[text_label]
                text = item['text']
                new_data.append({"label": label, "text": text})

    save_path = in_path.replace(".pickle", "_mapped.pickle")
    with open(save_path, 'wb') as handle:
        pickle.dump(new_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def process_twitter_covid_q6():
    in_path = "/home/nayeon/misinfo/data/preprocessed/twitter_q6.pickle"

    new_data = []
    with open(in_path, 'rb') as handle:
        data = pickle.load(handle)
        for item in data:
            if "yes" in item['label']:
                new_data.append({"label": "harmful", "text": item['text']})
            else:
                new_data.append({"label": "not_harmful", "text": item['text']})
        
    save_path = in_path.replace(".pickle", "_mapped.pickle")
    with open(save_path, 'wb') as handle:
        pickle.dump(new_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def process_twitter_covid_q7():
    in_path = "/home/nayeon/misinfo/data/preprocessed/twitter_q7.pickle"

    new_data = []
    with open(in_path, 'rb') as handle:
        data = pickle.load(handle)
        for item in data:
            if "yes" in item['label']:
                new_data.append({"label": "attention", "text": item['text']})
            else:
                new_data.append({"label": "not_attention", "text": item['text']})
        
    save_path = in_path.replace(".pickle", "_mapped.pickle")
    with open(save_path, 'wb') as handle:
        pickle.dump(new_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    ## preslav's covid twitter datasets
    # preprocess_twitter_covid()
    # process_twitter_covid_q2()
    # process_twitter_covid_q6()
    process_twitter_covid_q7()

    # propaganda()

    # additional datasets
    # prepare_fakenewsnet_gossip()
    # prepare_fakenewsnet_politifact()
    # prepare_fakenewsnet_buzzfeed()

    # nlp = spacy.load("en_core_web_sm")
    # # Webis DATA
    # print("webis")
    # prepare_webis(nlp)
    # #
    # print("clickbait")
    # prepare_clickbait(nlp)


    ## Basil DATA
    # create_basil_data()

    ## LIAR DATA
    # liar dataset analysis
    # pos_tag_liar_dataset()

    # liar_tag_analysis()
    # liar_tag_analysis(phase_path='liar_dev_pos.jsonl')
    # liar_tag_analysis(phase_path='liar_test_pos.jsonl')

    ## FEVER DATA
    # data_len = {'train': 145449, 'dev': 19998}

    # # pos tagging
    # pos_tag_fever(filename='shared_task_test')
    # pos_tag_fever(filename='shared_task_dev')
    # pos_tag_fever(filename='train')

    # # splitting into verifiable and unverifiable
    # split_fever_by_verifiability(filename='shared_task_dev', total=data_len['dev'])
    # split_fever_by_verifiability(filename='train', total=data_len['train'])

    # random_sample_fever('shared_task_dev', data_len['dev'], 50)

    # create questions for BERT
    # 1. NE based template TODO


    # 2. brute force TODO

    # filename = 'shared_task_dev'
    # nei_eg = defaultdict(list)
    # list_of_ne_tags = ['PRODUCT', 'CARDINAL', 'NORP', 'FAC', 'TIME', 'ORDINAL', 'LAW', 'GPE', 'LANGUAGE', 'QUANTITY', 'PERSON', 'EVENT', 'PERCENT', 'LOC', 'ORG', 'MONEY', 'DATE', 'WORK_OF_ART']
    # with jsonlines.open("/private/home/nayeon7lee/misinfo_data/fever/{}_ne_verifiable.jsonl".format(filename)) as reader:
    #     for obj in tqdm(reader):
    #         ne = obj['nei']
    #         if len(ne) == 1:
    #             nei_eg[ne[0][1]].append((ne[0][0], obj['claim'], obj['label']))
    #         else:
    #             pass
    #             # for entity in ne:
    #             #     nei_eg[entity[1]].append((entity[0], obj['claim'], obj['label']))
    #
    # print(nei_eg['PERSON'])
    # # for key, value in nei_eg.items():
    # #     print(key)
    # #     print(len(value))


    # # Webis DATA
    # print("webis")
    # prepare_webis(nlp)
    #
    # print("clickbait")
    # prepare_clickbait(nlp)


    ## Basil DATA
    # create_basil_data()

    ## LIAR DATA
    # liar dataset analysis
    # pos_tag_liar_dataset()

    # liar_tag_analysis()
    # liar_tag_analysis(phase_path='liar_dev_pos.jsonl')
    # liar_tag_analysis(phase_path='liar_test_pos.jsonl')

    ## FEVER DATA
    # data_len = {'train': 145449, 'dev': 19998}

    # # pos tagging
    # pos_tag_fever(filename='shared_task_test')
    # pos_tag_fever(filename='shared_task_dev')
    # pos_tag_fever(filename='train')

    # # splitting into verifiable and unverifiable
    # split_fever_by_verifiability(filename='shared_task_dev', total=data_len['dev'])
    # split_fever_by_verifiability(filename='train', total=data_len['train'])

    # random_sample_fever('shared_task_dev', data_len['dev'], 50)

    # create questions for BERT
    # 1. NE based template TODO


    # 2. brute force TODO

    # filename = 'shared_task_dev'
    # nei_eg = defaultdict(list)
    # list_of_ne_tags = ['PRODUCT', 'CARDINAL', 'NORP', 'FAC', 'TIME', 'ORDINAL', 'LAW', 'GPE', 'LANGUAGE', 'QUANTITY', 'PERSON', 'EVENT', 'PERCENT', 'LOC', 'ORG', 'MONEY', 'DATE', 'WORK_OF_ART']
    # with jsonlines.open("/private/home/nayeon7lee/misinfo_data/fever/{}_ne_verifiable.jsonl".format(filename)) as reader:
    #     for obj in tqdm(reader):
    #         ne = obj['nei']
    #         if len(ne) == 1:
    #             nei_eg[ne[0][1]].append((ne[0][0], obj['claim'], obj['label']))
    #         else:
    #             pass
    #             # for entity in ne:
    #             #     nei_eg[entity[1]].append((entity[0], obj['claim'], obj['label']))
    #
    # print(nei_eg['PERSON'])
    # # for key, value in nei_eg.items():
    # #     print(key)
    # #     print(len(value))

    print("done")

