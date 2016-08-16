# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random shuffle
from random import shuffle

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression

from os import path
import logging
import sys
import math

log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

CORPUS_TEST = 'test'
CORPUS_TRAIN = 'train'

class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sentences = []
        self.corpus_resume = {
            CORPUS_TRAIN + '_correct' : 0,
            CORPUS_TRAIN + '_incorrect' : 0,
            CORPUS_TEST + '_correct' : 0,
            CORPUS_TEST + '_incorrect' : 0,
        }
        self.__prepare(sources)

    def __prepare(self, sources):
        corpus = []

        for source in sources: # correct and incorrect
            prefix = path.splitext(source)[0]
            num_train = 0
            num_test = 0
            for it in range(0, 10): # increasing the corpus :)
                with utils.smart_open(source) as fin:
                    lines = fin.readlines()
                    shuffle(lines)

                    tr = []
                    te = []

                    for item_no, line in enumerate(lines): # lines of file
                        for_testing = int(math.ceil(len(lines) * 0.10)) # 10% for testing
                        te = lines[:for_testing]
                        tr = lines[for_testing:]

                    self.corpus_resume[CORPUS_TRAIN + '_%s' % prefix] = self.corpus_resume[CORPUS_TRAIN + '_%s' % prefix] + len(tr)
                    for t in tr:
                        self.sentences.append(
                            TaggedDocument(utils.to_unicode(t).split(), ['%s_%s_%s' % (CORPUS_TRAIN, prefix, num_train)])
                        )
                        num_train = num_train +1
                    self.corpus_resume[CORPUS_TEST + '_%s' % prefix] = self.corpus_resume[CORPUS_TEST + '_%s' % prefix] + len(te)
                    for t in te:
                        self.sentences.append(
                            TaggedDocument(utils.to_unicode(t).split(), ['%s_%s_%s' % (CORPUS_TEST, prefix, num_test)])
                        )
                        num_test = num_test +1

    def __iter__(self):
        for sentence in self.sentences:
            yield sentence

    def to_array(self):
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

    def get_amount(self, key):
        return self.corpus_resume[key]
    

log.info('source load')
sources = ['correct.txt', 'incorrect.txt']

log.info('TaggedDocument')
sentences = TaggedLineSentence(sources)

TRAIN_SIZE = sentences.get_amount(CORPUS_TRAIN + '_correct') + sentences.get_amount(CORPUS_TRAIN + '_incorrect')
TEST_SIZE = sentences.get_amount(CORPUS_TEST + '_correct') + sentences.get_amount(CORPUS_TEST + '_incorrect')

log.info('D2V')
DOC2VEC_SIZE = 400
model = Doc2Vec(min_count=1, window=1, size=DOC2VEC_SIZE, negative=5, workers=7)
model.build_vocab(sentences.to_array())

log.info('Epoch')
for epoch in range(10):
	log.info('EPOCH: {}'.format(epoch))
	model.train(sentences.sentences_perm())

log.info('Model Save')
model.save('./sfta.d2v')
model = Doc2Vec.load('./sfta.d2v')

log.info('Scoring')
train_arrays = numpy.zeros((TRAIN_SIZE, DOC2VEC_SIZE))
train_labels = numpy.zeros(TRAIN_SIZE)

for i in range(sentences.get_amount(CORPUS_TRAIN + '_correct')):
    prefix_train_correct = 'train_correct_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_correct]
    train_labels[i] = 1

for i in range(sentences.get_amount(CORPUS_TRAIN + '_incorrect')):
    prefix_train_incorrect = 'train_incorrect_' + str(i)
    train_arrays[sentences.get_amount(CORPUS_TRAIN + '_correct') + i] = model.docvecs[prefix_train_incorrect]
    train_labels[sentences.get_amount(CORPUS_TRAIN + '_correct') + i] = 0    

test_arrays = numpy.zeros((TEST_SIZE, DOC2VEC_SIZE))
test_labels = numpy.zeros(TEST_SIZE)

for i in range(sentences.get_amount(CORPUS_TEST + '_correct')):
    prefix_test_correct = 'test_correct_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_correct]
    test_labels[i] = 1

for i in range(sentences.get_amount(CORPUS_TEST + '_incorrect')):
    prefix_test_incorrect = 'test_incorrect_' + str(i)
    test_arrays[sentences.get_amount(CORPUS_TEST + '_correct') + i] = model.docvecs[prefix_test_incorrect]
    test_labels[sentences.get_amount(CORPUS_TEST + '_correct') + i] = 0     

log.info('Fitting')
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

#LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
print '================================================='
print "total: {}".format(len(sentences.sentences))
print "training: {}".format(len(train_arrays))
print "testing: {}".format(len(test_arrays))
print "accuarcy: {}%".format((classifier.score(test_arrays, test_labels) * 100))
print '================================================='
