import re
from re import finditer
from math import log
from numpy import zeros, abs, argsort
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


def key_sen_gen(content):
    # Clean text to make a list of sentences
    sent_lst = clean_txt(content)

    # Remove stop words and construct bag of words for each sentence
    bags = gen_bag_of_words(sent_lst)

    # Generating IDs, build the sparse matrix a from the bag-of-words representation
    all_words = set()
    for b in bags:
        all_words |= b
    w_t_id = {w: k for k, w in enumerate(all_words)}
    id_t_w = {k: w for k, w in enumerate(all_words)}

    rows, cols, vals = gen_coords(bags, w_t_id)

    a = csr_matrix((vals, (rows, cols)), shape=(len(w_t_id), len(bags)))

    # Calculate SVD
    sigma, u, v = get_svds_largest(a)

    # Rank words
    word_ranking = rank_words(u)
    words = [id_t_w[k] for k in word_ranking[:5]]

    # Rank sentences
    sentence_ranking = rank_sentences(v)
    sentences = [sent_lst[k] for k in sentence_ranking[:5]]

    return words, sentences


def clean_txt(text):
    sentences = []
    for line in text.splitlines():
        line_sentences = [st.strip() for st in re.split('[.?!]', line)]
        sentences += [sen for sen in line_sentences if sen != '']
    return sentences


def clean_sent(single_sent):
    pattern = r"[a-z]+('[a-z])?[a-z]*"
    return [match.group(0) for match in finditer(pattern, single_sent.lower())]


def gen_bag_of_words(sent):
    assert isinstance(sent, list)

    return [bag(s) for s in sent]


def bag(s):
    lemmatizer = WordNetLemmatizer()
    temp_bag = []
    for word in clean_sent(s):
        for i, j in pos_tag([word]):
            if j[0].lower() in ['a', 'n', 'v']:
                temp_bag.append(
                    lemmatizer.lemmatize(word, j[0].lower()))
            else:
                temp_bag.append(lemmatizer.lemmatize(word))
    return set(temp_bag) - set(stopwords.words('english'))


def gen_coords(bags, word_to_id):
    m, n = len(word_to_id.keys()), len(bags)
    row, col, val = [], [], []
    # Compute the `n_i` values
    num_docs = zeros(m)
    for ba in bags:
        for w in ba:
            num_docs[word_to_id[w]] += 1

    # Construct a coordinate representation
    for j, ba in enumerate(bags):  # loop over each sentence j
        for w in ba:  # loop over each word i in sentence j
            i = word_to_id[w]
            a_ij = 1.0 / log((n + 1) / num_docs[i])
            row.append(i)
            col.append(j)
            val.append(a_ij)

    return row, col, val


def get_svds_largest(a):
    u1, s1, v1 = svds(a, k=1, which='LM',
                      return_singular_vectors=True)
    return s1, abs(u1.reshape(a.shape[0])), abs(v1.reshape(a.shape[1]))


def rank_words(u0):
    return argsort(u0)[::-1]


def rank_sentences(v0):
    return argsort(v0)[::-1]
