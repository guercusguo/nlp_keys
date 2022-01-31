import yake

def add_yake_key_1 (text):
    language = "en"
    max_ngram_size = 1
    deduplication_thresold = 0.9
    deduplication_algo = 'seqm'
    windowSize = 10
    numOfKeywords = 20
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
    yake_keywords = custom_kw_extractor.extract_keywords(text)
    for kw in keywords:
        return str(yake_keywords)

def add_yake_key_2 (text):
    language = "en"
    max_ngram_size = 2
    deduplication_thresold = 0.9
    deduplication_algo = 'seqm'
    windowSize = 10
    numOfKeywords = 6
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
    yake_keywords = custom_kw_extractor.extract_keywords(text)
    for kw in keywords:
        return str(yake_keywords)

def add_yake_key_3 (text):
    language = "en"
    max_ngram_size = 3
    deduplication_thresold = 0.9
    deduplication_algo = 'seqm'
    windowSize = 10
    numOfKeywords = 10
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
    yake_keywords = custom_kw_extractor.extract_keywords(text)
    for kw in keywords:
        return str(yake_keywords)
lectures['yake_keys_1'] = np.vectorize(add_yake_key_1)(lectures['text'])
lectures['yake_keys_2'] = np.vectorize(add_yake_key_2)(lectures['text'])
lectures['yake_keys_3'] = np.vectorize(add_yake_key_3)(lectures['text'])

#lectures['yake_keys_1'] = lectures['yake_keys_1'].apply(lambda x: ','.join(set(x.split(','))))
#lectures['yake_keys_1'] = lectures.yake_keys_1.apply(lambda x: x[1:-1].split(','))
#lectures['yake_keys_1'] = lectures['yake_keys_1'].apply(lambda x: len(str(x).split(",")))
lectures.head()
lectures.describe()
lectures.to_csv('lectures_tokens_yake.csv',sep='\t') # custom delimiter (tab)
