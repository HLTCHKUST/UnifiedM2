


# Add additional Misinfo tasks
misinfo_tasks = ['liar', 'webis', 'clickbait', 'basil_detection', 'basil_type', 'basil_polarity', 
                'fever', 'fever_binary', 'rumour_detection', 'rumour_veracity', 'rumour_veracity_binary',
                'fnn_politifact', 'fnn_buzzfeed', 'fnn_gossip',  'fnn_buzzfeed_title', 'propaganda', 
                'covid_twitter_q1', 'fnn_politifact_title', 'covid_twitter_q2', 'rumour_binary_byEvents' ]


LABELS = {
    "rumour_detection": {'rumour': 1, 'non-rumour': 0},
    "rumour_veracity": {'false': 0, 'true': 1, 'unverified': 2},
    "rumour_veracity_binary": {'false': 1, 'true': 0},
    "rumour_binary_byEvents": {'false': 1, 'true': 0},
    "stance": {'s': 0, 'c': 1, 'd': 2, 'q': 3,
                        '0': 's', '1': 'c', '2': 'd', '3': 'q'},
    "s2s": {"support": "s", "deny": "d", "query": "q", "comment": "c",
                        "agreed": "s", "disagreed": "d", "appeal-for-more-information": "q",
                        "supporting": "s", "denying": "d", "undersp8856ecified": "q"},
    # "liar": {'half-true': 1, 'false': 0, 'mostly-true': 2, 'barely-true': 1, 'true': 2, 'pants-fire': 0},
    "liar": {'half-true': 0, 'false': 1, 'mostly-true': 2, 'barely-true': 3, 'true': 4, 'pants-fire': 5},
    "webis": {'mostly true': 0,
         'no factual content': 1,
         'mixture of true and false': 1,
         'mostly false': 1},
    "clickbait": {"no-clickbait": 0, "clickbait": 1, "0": "no-clickbait", "1": "clickbait"},
    "basil_detection": {'no-bias': 0, 'Informational': 1, 'Lexical': 1, 'both': 1},
    # "basil_detection": {'no-bias': 0, 'contain-bias': 1},
    "basil_type": {"Lexical": 0, "Informational": 1},
    "basil_polarity": {"Negative": 0, "Positive": 1},
    "fever": {'REFUTES': 0, 'SUPPORTS': 1, 'NOT ENOUGH INFO': 2},
    "fever_binary": {'REFUTES': 0, 'SUPPORTS': 1},
    'fnn_politifact': {"real": 0, "fake": 1},
    'fnn_buzzfeed': {"real": 0, "fake": 1},
    'fnn_gossip': {"real": 0, "fake": 1},
    'fnn_buzzfeed_title': {"real": 0, "fake": 1},
    'propaganda': {"no_propaganda": 0, "has_propaganda": 1},
    'covid_twitter_q1': {"yes": 0, "no": 1},
    'fnn_politifact_title': {"real": 0, "fake": 1},
    'covid_twitter_q2': {"no_false": 0, "contains_false": 1},
}

task2idx = {
    'liar': 0,
    'webis': 1,
    'clickbait': 2,
    'basil_detection': 3,
    'basil_type': 4,
    'basil_polarity': 5,
    'fever': 6,
    'fever_binary': 7,
    'rumour_detection': 8,
    'rumour_veracity': 9,
    'rumour_veracity_binary': 10,
    'fnn_politifact': 11,
    'fnn_buzzfeed': 12,
    # 'fnn_gossip': 13,
    'fnn_buzzfeed_title': 14,
    'propaganda': 15,
    'covid_twitter_q1': 16,
    'fnn_politifact_title': 17,

    'covid_twitter_q2': 18,
    'rumour_binary_byEvents': 21
}
