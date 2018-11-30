"""
NLTK is a Python package for NLP. Please see here http://www.nltk.org/ for downloading
and documentation. You can download the Brown corpus using nltk.download(’brown’) and traverse
its sentences using the tagged sents() method. Please refer to the NLTK documentation for further
information. Please use NLTK just for loading and reading the corpus.
"""
import nltk

from nltk.corpus import brown
import operator
import re

class My_ex2_class:
    # static variables
    category = "news"
    training_portion = 0.9
    unknown_tag = 'NN'
    start = 'START'
    stop = 'STOP'
    add = 0
    rare_words_threshold = 5
    ZERO_PROB = 0

    # common lists/dicts
    training_set = []
    training_words = []
    test_words = []
    test_set = []
    argmax_tag_per_word = {}
    unknown_words = []
    tag_to_tag_dict = {}
    tag_to_tag_prob = {}
    tag_to_word_dict = {}
    training_tags = []
    tag_freqs = {}
    pseudo_words = {}
    rare_words = []
    pseudo_options = {
        # Bikel et. al
        "^\d{2}$": "twoDigitNum",
        "^\d{4}$": "fourDigitNum",
        "[A-Z](\d)+": "containsDigitAndAlpha",
        "^(\d+)-(\d+)$": "containsDigitAndDash",
        "^(\d+/\d+)+(/\d+)*$": "containsDigitAndSlash",
        "^(\d+,\d+)+$": "containsDigitAndComma",
        "^(\d+\.\d+)+$": "containsDigitAndPeriod",
        "^\d+$": "othernum",
        "^[A-Z]+$": "allCaps",
        "^[A-Z][a-z]{0,2}\\.$": "capPeriod",
        "[A-Z][a-z]+": "firstWord",
        "^[A-Z][a-zA-Z]+$": "initCap",
        "^[a-z]+$": "lowercase",
        # some additional
        "[A-Za-z0-9]*\\-[A-Za-z0-9]*": "hyphenSeparated",
        "[A-Za-z0-9]*'s": "related",
        ".*": "other"
    }

    def __init__(self):
        return

    """
    (a) Use the NLTK toolkit for importing the Brown corpus. This corpus contains text from 500
    sources, and the sources have been categorized by genre. Here we will use a portion of the corpus:
    the “news” category.
    1. Load the tagged sentences for this portion of the corpus.
    2. Then, divide the obtained corpus into training set and test set such that the test set is formed
        by the last 10% of the sentences.
    """

    def initialize_corpus(self, is_pseudo = 0):
        news_sents = brown.tagged_sents(categories=self.category)
        list(filter(None.__ne__, news_sents))
        if is_pseudo == 1:
            for i, s in enumerate(news_sents):
                for j, w in enumerate(s):
                    if w[0] in self.rare_words:
                        lst = list(news_sents[i][j])
                        lst[0] = self.swap_with_pseudo(w[0])
                        news_sents[i][j] = tuple(lst)
        return news_sents

    def get_corpus_data(self, new_sents, is_pseudo = 0):
        # divide to training and test sets
        for i, s in enumerate(news_sents):
            if i < len(news_sents) * self.training_portion:
                self.training_set.append(s)
                for w_tuple in s:
                    self.training_words.append(w_tuple[0])
                    self.training_tags.append(w_tuple[1])
            if i >= len(news_sents) * self.training_portion:
                self.test_set.append(s)
                for w_tuple in s:
                    if is_pseudo:
                        self.test_words.append(w_tuple[0])
                    else:
                        self.test_words.append(w_tuple[0])

        # remove duplicates
        self.training_words = list(set(self.training_words))
        self.test_words = list(set(self.test_words))
        self.training_tags = list(set(self.training_tags)) + [self.start, self.stop]
        # lists of unknown words
        self.unknown_words = list(set(self.test_words) - set(self.training_words))

    """
    (b) Implementation of the most likely tag baseline
        i.  Using the training set, compute for each word the tag that maximizes p(tag|word). Assume
            that the most likely tag of all the unknown words is “NN”. (Unknown words are words that
            appear in the test set but not in the training set.)
        ii. Using the test set, compute the error rate (i.e., 1−accuracy) for known words and for unknown
            words, as well as the total error rate
    """

    def calc_tag_freqs(self, is_pseudo=0):
        self.tag_freqs = {}
        # init dictionary
        for s in self.training_set:
            for w in s:
                self.tag_freqs[w[0]] = {}

        # init tag appearances
        for s in self.training_set:
            for w in s:
                self.tag_freqs[w[0]][w[1]] = 0

        # count tag appearances per word
        for s in self.training_set:
            for w in s:
                self.tag_freqs[w[0]][w[1]] += 1

    def compute_argmax_tag(self):
        argmax_tag_per_word = {}

        for w in self.tag_freqs:
            argmax_tag_per_word[w] = max(self.tag_freqs[w].items(), key=operator.itemgetter(1))[0]

        return argmax_tag_per_word

    def calc_accuracy(self, is_pseudo = 0):
        training_max_per_w = self.compute_argmax_tag()
        # test_max_per_w = self.compute_argmax_tag(self.test_set)
        tot_kn_cnt = 0
        correct_kn_cnt = 0
        tot_unk_cnt = 0
        correct_unk_cnt = 0

        for s in self.test_set:
            for w in s:
                w0 = w[0]
                if w0 in training_max_per_w:
                    tot_kn_cnt += 1
                    if w[1] == training_max_per_w[w0]:
                        correct_kn_cnt += 1
                else:
                    tot_unk_cnt += 1
                    if w[1] == self.unknown_tag:
                        correct_unk_cnt += 1

        # calc accuracy: <known %, unknown %, total %>
        return correct_kn_cnt / tot_kn_cnt, correct_unk_cnt / tot_unk_cnt, (correct_kn_cnt + correct_unk_cnt) / (
                tot_kn_cnt + tot_unk_cnt)

    """
    (c) Implementation of a bigram HMM tagger
    i. Training phase: Compute the transition and emission probabilities of a bigram HMM tagger
    directly on the training set using maximum likelihood estimation.
    ii. Implement the Viterbi algorithm corresponding to the bigram HMM model. (Choose an
    arbitrary tag for unknown words.)
    iii. Run the algorithm from c)ii) on the test set. Compute the error rates and compare to the
    results from b)ii).
    """

    def init_trans_emis(self, is_pseudo = 0):
        curr_tag = None
        # create a dict. of tags and their following tags
        self.tag_to_tag_dict = {tag: [] for tag in self.training_tags}
        for s in self.training_set:
            for w, t in s:
                if curr_tag is None:
                    curr_tag = t
                    self.tag_to_tag_dict[self.start].append(curr_tag)
                    continue
                else:
                    self.tag_to_tag_dict[curr_tag].append(t)
                    curr_tag = t
            self.tag_to_tag_dict[s[len(s)-1][1]].append(self.stop)
            curr_tag = None

        # create a dict. of tags and their corresponding words
        self.tag_to_word_dict = {tag:[] for tag in self.training_tags}
        for s in self.training_set:
            for w,t in s:
                self.tag_to_word_dict[t].append(w)
        self.tag_to_word_dict[self.start] = [self.start]
        self.tag_to_word_dict[self.stop] = [self.stop]

        self.init_tag_to_tag_prob()

    def init_tag_to_tag_prob(self):
        self.tag_to_tag_prob = {}
        for v in self.training_tags:
            self.tag_to_tag_prob[v] = {}
            for u in self.training_tags:
                self.tag_to_tag_prob[v][u] = self.calc_cond_prob(self.tag_to_tag_dict, v, u)

    def viterbi(self, sent_words, is_pseudo = 0):
        n = len(sent_words)
        max_table = [{} for i in range(n + 1)]
        bp_table = [{} for i in range(n + 1)]
        sent_tags = ["" for i in range(n)]
        r_vals = {}

        if is_pseudo:
            sent_words = [w for w in sent_words]

        # init max_table
        for v in self.training_tags:
            max_table[0][v] = 0
        max_table[0][self.start] = 1

        # dynamic algorithm to fill tables
        for k in range(1, n + 1):
            # print("k = " + str(k))
            for v in self.training_tags:
                tag_to_word_prob = self.calc_cond_prob(self.tag_to_word_dict, sent_words[k - 1], v, self.add)
                if tag_to_word_prob:
                    # no need to multiply by tag_to_word_prob now, do it later
                    r_vals = {u: (max_table[k - 1][u] * self.tag_to_tag_prob[v][u]) for u in self.training_tags}
                else:
                    r_vals = {self.unknown_tag: 0}
                bp_table[k - 1][v] = max(r_vals, key=r_vals.get)
                max_table[k][v] = tag_to_word_prob * r_vals[bp_table[k - 1][v]]

        # find tags by backtracking
        r_vals = {v: (max_table[n][v] * self.tag_to_tag_prob[self.stop][v]) for v in self.training_tags}
        sent_tags[n - 1] = max(r_vals, key=r_vals.get)

        for k in range(n - 1, 0, -1):
            sent_tags[k - 1] = bp_table[k][sent_tags[k]]

        return sent_tags

    def calc_accuracy_viterbi(self, is_pseudo=0):
        tot_known = 0
        corr_known = 0
        tot_unk = 0
        corr_unk = 0
        for s in self.test_set:
            s_words = [w for w,t in s]
            sent_tags = self.viterbi(s_words, is_pseudo)
            for i,w in enumerate(s):
                if w[0] not in self.unknown_words:
                    tot_known += 1
                    if w[1] == sent_tags[i]:
                        corr_known += 1
                else:
                    tot_unk += 1
                    if w[1] == self.unknown_tag:
                        corr_unk += 1
        return corr_known / tot_known, corr_unk / tot_unk, (corr_known + corr_unk) / (tot_known + tot_unk)

    """
    (d) Using Add-one smoothing
        i.  Training phase: Compute the emission probabilities of a bigram HMM tagger directly on the
            training set using (Laplace) Add-one smoothing.
        ii. Using the new probabilities, run the algorithm from c)ii) on the test set. Compute the error
            rates and compare to the results from b)ii) and c)iii).
    """
    def set_add_laplace(self, param):
        self.add = param

    def calc_cond_prob(self, dict, x, y, add=0):
        """calculates p(x|y) for any dictionary of arrays {key:[values]}"""
        if (y not in dict.keys()) or (len(dict[y]) == 0):
            self.ZERO_PROB += 1
            return 0.0
        return (dict[y].count(x) + add) / (len(dict[y]) + len(set(dict[y])) * add)

    """
    (e) Using pseudo-words
        i.   Design a set of pseudo-words for unknown words in the test set and low-frequency words in
             the training set.
        ii.  Using the pseudo-words as well as maximum likelihood estimation (as in c)i)), run the Viterbi
             algorithm on the test set. Compute the error rates and compare to the results from b)ii),
             c)iii) and d)ii).
        iii. Using the pseudo-words as well as Add-One smoothing (as in d)i)), run the Viterbi algorithm
             on the test set. Compute the error rates and compare to the results from b)ii), c)iii), d)ii)
             and e)ii). For the results obtained using both pseudo-words and Add-One smoothing, build a
             confusion matrix and investigate the most frequent errors. A confusion matrix is an |K| over
             |K| matrix, where the (i, j) entry corresponds to the number of tokens which have a true tag
             i and a predicted tag j.
    """

    def find_rare_words(self):
        # find low frequency words
        for w in self.tag_freqs:
            curr_cnt = 0
            for t in self.tag_freqs[w]:
                curr_cnt += self.tag_freqs[w][t]
            if curr_cnt < self.rare_words_threshold:
                self.rare_words.append(w)

    def swap_with_pseudo(self, word):
        for opt in self.pseudo_options:
            r = re.compile(opt)
            if r.match(word):
                return self.pseudo_options[opt]
        return ""


if __name__ == '__main__':

    print("========Ex 2 Output Starts Here========")
    my_class = My_ex2_class()
    print("1. initializing corpus...")
    news_sents = my_class.initialize_corpus(0)
    print("2. getting corpus data...")
    my_class.get_corpus_data(news_sents)
    print("3. calculating frequent tags...")
    my_class.calc_tag_freqs()
    print("4. calculating accuracy...")
    ba, bb, bc = my_class.calc_accuracy()
    print("(b) Known: " + str(1-ba), "\n\tUnknown: " + str(1-bb), "\n\tTotal: " + str(1-bc))
    print("Initializing dictionaries...")
    my_class.init_trans_emis()
    print("5. calculating accuracy with viterbi...")
    ca, cb, cc = my_class.calc_accuracy_viterbi()
    print("(c) Known: " + str(1-ca), "\n\tUnknown: " + str(1-cb), "\n\tTotal: " + str(1-cc))
    print("zero prob: "+str(my_class.ZERO_PROB))
    my_class.ZERO_PROB = 0
    print("Setting add to 1...")
    my_class.set_add_laplace(1)
    print("calculating accuracy with viterbi and add-one...")
    da, db, dc = my_class.calc_accuracy_viterbi()
    print("(d) Known: " + str(1-da), "\n\tUnknown: " + str(1-db), "\n\tTotal: " + str(1-dc))
    print("zero prob: " + str(my_class.ZERO_PROB))
    my_class.set_add_laplace(0)
    print("6. finding rare words...")
    my_class.find_rare_words()
    # rerun with pseudo words```
    print("Reinitializing corpus...")
    news_sents = my_class.initialize_corpus(1)
    print("Rerunning with pseudo-words...")
    my_class.get_corpus_data(news_sents, 1)
    print("Calculating tag freqs...")
    my_class.calc_tag_freqs(1)
    print("Reinitializing dictionaries...")
    my_class.init_trans_emis(1)
    print("Calculating Viterbi accuracy...")
    e1a, e1b, e1c = my_class.calc_accuracy_viterbi(1)
    print("(e2) Known: " + str(1-e1a), "\n\tUnknown: " + str(1-e1b), "\n\tTotal: " + str(1-e1c))
    my_class.set_add_laplace(1)
    e2a, e2b, e2c = my_class.calc_accuracy_viterbi(1)
    print("(e3) Known: " + str(1-e2a), "\n\tUnknown: " + str(1-e2b), "\n\tTotal: " + str(1-e2c))