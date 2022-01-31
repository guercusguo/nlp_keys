from rake_nltk import Rake
r = Rake(language='english')
r = Rake(
    stopwords=['has', 'ours', 'needn', 'through', 'such', 'for', 'she', 'in', "won't", "you've", "mightn't", 'o', 'between', 'will', 'both', 'y', 'those', 'did', 'a', 'at', 'below', 'very', 'does', 'mightn', 'mustn', 'haven', 'over', 'than', 'same', 'until', 'been', 'is', 'ma', 'have', 'should', "shouldn't", 'only', 'itself', 'him', 'd', 'and', 'isn', "you'd", 'most', 'couldn', 'its', 'up', 'being', 'weren', 'further', 'why', 'The', 'how', 'down', 'am', 'hers', 'who', 'i', 'or', 'when', 'from', 'therefore', "she's", 'them', 'own', 'm', "wouldn't", 'where', 'if', 'with', 'we', 'ain', 'too', 'about', 'during', 'other', "that'll", 'whom', 'while', 'nor', 'having', "shan't", 'any', 'didn', 'so', 't', 'hadn', 'he', 'then', "isn't", 'which', "hadn't", 'these', 'against', "didn't", 'just', 'to', 'theirs', 'himself', 'that', 'doing', 'again', 'few', 'because', 'all', "doesn't", 'off', "weren't", 'yourselves', 'your', "hasn't", 'under', 'there', 'aren', 'doesn', 'hasn', "aren't", 'shouldn', 'their', "it's", 'don', 'wasn', 'the', 'her', 'they', 'themselves', 'it', 'In', 'once', 'our', 'here', 're', 'this', "wasn't", 'no', "needn't", 'out', 'before', 'yours', 'into', 'what', 'above', 'on', 'was', 'some', 'll', 'more', 'each', 've', "couldn't", 'yourself', 'had', 'his', 'not', 'myself', 'of', 'as', 'me', 'wouldn', 'an', 'you', "don't", 'case', "haven't", "you're", 'can', 'now', 'after', 'shan', 'my', 'but', "you'll", 'were', "should've", 'are', 'be', 'do', 's', 'herself', 'ourselves', 'by', "mustn't", 'won'],
    min_length=2,
    max_length=4
)
text = lectures.iloc[12]['text']
print str(r)
import RAKE
# https://csurfer.github.io/rake-nltk/_build/html/index.html
import operator
filepath = "keyword_extraction.txt"
#https://medium.com/datadriveninvestor/rake-rapid-automatic-keyword-extraction-algorithm-f4ec17b2886c
rake_object = RAKE.rake(filepath)
text = ""
text = lectures.iloc[45]['text']
keywords = rake_object.run(text)
print ('Keywords:', keywords)
wordscores = rake.calculate_word_scores(phraseList)
keywordcandidates = rake.generate_candidate_keyword_scores(phraseList, wordscores)
sortedKeywords = sorted(keywordcandidates.iteritems(),key=operator.itemgetter(1), reverse=True)
totalKeywords = len(sortedKeywords)for keyword in sortedKeywords[0:(totalKeywords / 3)]:
      print “Keyword: “, keyword[0], “, score: “, keyword[1]
