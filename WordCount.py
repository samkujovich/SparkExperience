# **Word Count Lab: Building a word count application**
# *Part 1:* Creating a base RDD and pair RDDs
# *Part 2:* Counting with pair RDDs
# *Part 3:* Finding unique words and a mean value
# *Part 4:* Apply word count to a file

# ** Part 1: Creating a base RDD and pair RDDs **

# ** Create a base RDD **
wordsList = ['cat', 'elephant', 'rat', 'rat', 'cat']
wordsRDD = sc.parallelize(wordsList, 4)
# Print out the type of wordsRDD
print type(wordsRDD)


# ** Pluralize and test **
def makePlural(word):
    """Adds an 's' to `word`.

    Note:
        This is a simple function that only adds an 's'.  No attempt is made to follow proper
        pluralization rules.

    Args:
        word (str): A string.

    Returns:
        str: A string with 's' added to it.
    """
    return word + 's'

print makePlural('cat')

# Load in the testing code
from test_helper import Test

# TEST Pluralize and test
Test.assertEquals(makePlural('rat'), 'rats', 'incorrect result: makePlural does not add an s')


# ** Apply `makePlural` to the base RDD **

pluralRDD = wordsRDD.map(lambda x: (makePlural(x)))
print pluralRDD.collect()


# TEST Apply makePlural to the base RDD
Test.assertEquals(pluralRDD.collect(), ['cats', 'elephants', 'rats', 'rats', 'cats'],
                  'incorrect values for pluralRDD')


# ** Pass a `lambda` function to `map` **
pluralLambdaRDD = wordsRDD.map(makePlural)
print pluralLambdaRDD.collect()

# TEST Pass a lambda function to map
Test.assertEquals(pluralLambdaRDD.collect(), ['cats', 'elephants', 'rats', 'rats', 'cats'],
                  'incorrect values for pluralLambdaRDD (1d)')


# ** (1e) Length of each word **
pluralLengths = (pluralRDD.map(lambda x: (len(x))).collect())
print pluralLengths

# TEST Length of each word
Test.assertEquals(pluralLengths, [4, 9, 4, 4, 4],
                  'incorrect values for pluralLengths')


# ** Pair RDDs **
wordPairs = wordsRDD.map(lambda x: (x,1))
print wordPairs.collect()

# TEST Pair RDDs
Test.assertEquals(wordPairs.collect(),
                  [('cat', 1), ('elephant', 1), ('rat', 1), ('rat', 1), ('cat', 1)],
                  'incorrect value for wordPairs')


# ** Part 2: Counting with pair RDDs **

# ** `groupByKey()` approach **
wordsGrouped = wordPairs.groupByKey()
for key, value in wordsGrouped.collect():
    print '{0}: {1}'.format(key, list(value))


# TEST groupByKey() approach
Test.assertEquals(sorted(wordsGrouped.mapValues(lambda x: list(x)).collect()),
                  [('cat', [1, 1]), ('elephant', [1]), ('rat', [1, 1])],
                  'incorrect value for wordsGrouped')


# ** Use `groupByKey()` to obtain the counts **

wordCountsGrouped = wordPairs.groupByKey().mapValues(len)
print wordCountsGrouped.collect()

# TEST Use groupByKey() to obtain the counts
Test.assertEquals(sorted(wordCountsGrouped.collect()),
                  [('cat', 2), ('elephant', 1), ('rat', 2)],
                  'incorrect value for wordCountsGrouped')


# ** Counting using `reduceByKey` **
from operator import add
wordCounts = wordPairs.reduceByKey(add)
print wordCounts.collect()

# TEST Counting using reduceByKey
Test.assertEquals(sorted(wordCounts.collect()), [('cat', 2), ('elephant', 1), ('rat', 2)],
                  'incorrect value for wordCounts')


# ** All together **
from operator import add
wordCountsCollected = wordsRDD.map(lambda x: (x,1)).reduceByKey(add).collect()
print wordCountsCollected

# TEST All together
Test.assertEquals(sorted(wordCountsCollected), [('cat', 2), ('elephant', 1), ('rat', 2)],
                  'incorrect value for wordCountsCollected')


# ** Finding unique words and a mean value **

from operator import add
wordCounts = wordPairs.reduceByKey(add)
uniqueWords = wordCounts.count()
print uniqueWords

# TEST Unique words
Test.assertEquals(uniqueWords, 3, 'incorrect count of uniqueWords')


# ** Mean using `reduce` **

from operator import add
totalCount = (wordCounts.map(lambda s: s[1]).reduce(add))
average = totalCount / float(uniqueWords)
print totalCount
print round(average, 2)

# TEST Mean using reduce
Test.assertEquals(round(average, 2), 1.67, 'incorrect value of average')


# ** Apply word count to a file **
from operator import add
def wordCount(wordListRDD):
    """Creates a pair RDD with word counts from an RDD of words.
   Args:
        wordListRDD (RDD of str): An RDD consisting of words.
    Returns:
        RDD of (str, int): An RDD consisting of (word, count) tuples."""
    _wordListRDD = wordListRDD.map(lambda x: (x,1)).reduceByKey(add)
    return _wordListRDD 
    
print wordCount(wordsRDD).collect()

# TEST wordCount function
Test.assertEquals(sorted(wordCount(wordsRDD).collect()),
                  [('cat', 2), ('elephant', 1), ('rat', 2)],
                  'incorrect definition for wordCount function')


# ** Capitalization and punctuation **
import re
def removePunctuation(text):
    """Removes punctuation, changes to lower case, and strips leading and trailing spaces.

    Note:
        Only spaces, letters, and numbers should be retained.  Other characters should should be
        eliminated (e.g. it's becomes its).  Leading and trailing spaces should be removed after
        punctuation is removed.

    Args:
        text (str): A string.

    Returns:
        str: The cleaned up string.
    """
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punc = ""
    for char in text:
        if char not in punctuations:
           no_punc = no_punc + char
    no_punc = no_punc.lower()  
    no_punc= no_punc.strip()
    return no_punc
print removePunctuation('Hi, you!')
print removePunctuation(' No under_score!')

# TEST Capitalization and punctuation
Test.assertEquals(removePunctuation(" The Elephant's 4 cats. "),
                  'the elephants 4 cats',
                  'incorrect definition for removePunctuation function')


# ** Load a text file **
import os.path
baseDir = os.path.join('data')
inputPath = os.path.join('cs100', 'lab1', 'shakespeare.txt')
fileName = os.path.join(baseDir, inputPath)

shakespeareRDD = (sc
                  .textFile(fileName, 8)
                  .map(removePunctuation))
print '\n'.join(shakespeareRDD
                .zipWithIndex()  # to (line, lineNum)
                .map(lambda (l, num): '{0}: {1}'.format(num, l))  # to 'lineNum: line'
                .take(15))


# ** Words from lines **
shakespeareWordsRDD = shakespeareRDD.flatMap(lambda x : x.split(' '))
shakespeareWordCount = shakespeareWordsRDD.count()
print shakespeareWordsRDD.top(5)
print shakespeareWordCount

# TEST Words from lines
Test.assertTrue(shakespeareWordCount == 927631 or shakespeareWordCount == 928908,
                'incorrect value for shakespeareWordCount')
Test.assertEquals(shakespeareWordsRDD.top(5),
                  [u'zwaggerd', u'zounds', u'zounds', u'zounds', u'zounds'],
                  'incorrect value for shakespeareWordsRDD')


# ** Remove empty elements **
shakeWordsRDD = shakespeareRDD.flatMap(lambda x : x.split())
shakeWordCount = shakeWordsRDD.count()
print shakeWordCount


# TEST Remove empty elements
Test.assertEquals(shakeWordCount, 882996, 'incorrect value for shakeWordCount')


# ** Count the words **

top15WordsAndCounts = wordCount(shakeWordsRDD).takeOrdered(15,  lambda row : -row[1])
print '\n'.join(map(lambda (w, c): '{0}: {1}'.format(w, c), top15WordsAndCounts))

# TEST Count the words
Test.assertEquals(top15WordsAndCounts,
                  [(u'the', 27361), (u'and', 26028), (u'i', 20681), (u'to', 19150), (u'of', 17463),
                   (u'a', 14593), (u'you', 13615), (u'my', 12481), (u'in', 10956), (u'that', 10890),
                   (u'is', 9134), (u'not', 8497), (u'with', 7771), (u'me', 7769), (u'it', 7678)],
                  'incorrect value for top15WordsAndCounts')