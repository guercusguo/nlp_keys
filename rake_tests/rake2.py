import RAKE
import operator

# Reka setup with stopword directory
stop_dir = "SmartStoplist.txt"
rake_object = RAKE.Rake(stop_dir)

# Sample text to test RAKE
text = text = lectures.iloc[1]['text']
# Extract keywords
keywords = rake_object.run(text, minCharacters = 3, maxWords = 3, minFrequency =4)
print ("keywords: ", keywords)

sentenceList = Rake.split_sentences(text)
stopwordpattern = Rake.build_stop_word_regex(filepath)
phraseList = Rake.generate_candidate_keywords(sentenceList, stopwordpattern)

# Rake -> https://github.com/fabianvf/python-rake

def add_col_key_1_1 (text, stop_dir):
    # Rake setup with stopword directory
    rake_object = RAKE.Rake(stop_dir)
    # Extract keywords
    keywords = rake_object.run(text, minCharacters = 3, maxWords = 1, minFrequency = 1)
    return str(keywords)

def add_col_key_1_2 (text, stop_dir):
    # Rake setup with stopword directory
    rake_object = RAKE.Rake(stop_dir)
    # Extract keywords
    keywords = rake_object.run(text, minCharacters = 3, maxWords = 1, minFrequency = 2)
    return str(keywords)

def add_col_key_2 (text, stop_dir):
    # Rake setup with stopword directory
    rake_object = RAKE.Rake(stop_dir)
    # Extract keywords
    keywords = rake_object.run(text, minCharacters = 3, maxWords = 2, minFrequency = 1)
    return str(keywords)

def add_col_key_3 (text, stop_dir):
    # Rake setup with stopword directory
    rake_object = RAKE.Rake(stop_dir)
    # Extract keywords
    keywords = rake_object.run(text, minCharacters = 3, maxWords = 3, minFrequency = 1)
    return str(keywords)

lectures['rake_keys_1_1'] = np.vectorize(add_col_key_1_1)(lectures['text'],"SmartStoplist.txt")
lectures['rake_keys_1_2'] = np.vectorize(add_col_key_1_2)(lectures['text'],"SmartStoplist.txt")
lectures['rake_keys_2'] = np.vectorize(add_col_key_2)(lectures['text'],"SmartStoplist.txt")
lectures['rake_keys_3'] = np.vectorize(add_col_key_3)(lectures['text'],"SmartStoplist.txt")

lectures.head()
lectures.describe()

lectures.iloc[9]['rake_keys']
