from rake_nltk import Rake

# Uses stopwords for english from NLTK, and all puntuation characters by
# default
r = Rake('english')
# Extraction given the text.
r.extract_keywords_from_text(lectures.iloc[0]['text'])
# Extraction given the list of strings where each string is a sentence.
r.extract_keywords_from_sentences(lectures['text'])
# To get keyword phrases ranked highest to lowest.
r.get_ranked_phrases()
# To get keyword phrases ranked highest to lowest with scores.
r.get_ranked_phrases_with_scores()
