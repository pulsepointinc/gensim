#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from gensim.summarization.pagerank_weighted import pagerank_weighted as _pagerank
from gensim.summarization.textcleaner import clean_text_by_word as _clean_text_by_word
from gensim.summarization.textcleaner import tokenize_by_word as _tokenize_by_word
from gensim.summarization.commons import build_graph as _build_graph
from gensim.summarization.commons import remove_unreachable_nodes as _remove_unreachable_nodes
from gensim.parsing.preprocessing import preprocess_strings_list as _preprocess_strings
from gensim.parsing.preprocessing import DEFAULT2_FILTERS
from gensim.utils import to_unicode
from itertools import combinations as _combinations
from six.moves.queue import Queue as _Queue
from six.moves import xrange
from six import iteritems


WINDOW_SIZE = 2

"""
Check tags in http://www.clips.ua.ac.be/pages/mbsp-tags and use only first two letters
Example: filter for nouns and adjectives:
INCLUDING_FILTER = ['NN', 'JJ']
"""
INCLUDING_FILTER = ['NN', 'JJ']
EXCLUDING_FILTER = []


def _get_pos_filters():
    return frozenset(INCLUDING_FILTER), frozenset(EXCLUDING_FILTER)


def _get_words_for_graph(tokens, pos_filter):
    if pos_filter is None:
        include_filters, exclude_filters = _get_pos_filters()
    else:
        include_filters = set(pos_filter)
        exclude_filters = frozenset([])
    if include_filters and exclude_filters:
        raise ValueError("Can't use both include and exclude filters, should use only one")

    result = []
    for word, unit in iteritems(tokens):
        # print word, unit.tag
        if exclude_filters and unit.tag in exclude_filters:
            continue
        if (include_filters and unit.tag in include_filters) or not include_filters or not unit.tag:
            # print "Included:", word, unit.token
            result.append(unit.token)
    # print len(result)
    # print result
    return result


def _get_first_window(split_text):
    return split_text[:WINDOW_SIZE]


def _set_graph_edge(graph, tokens, word_a, word_b):
    # print "Check Edge:", word_a, word_b,
    if word_a in tokens and word_b in tokens:
        lemma_a = tokens[word_a].token
        lemma_b = tokens[word_b].token
        edge = (lemma_a, lemma_b)
        if graph.has_node(lemma_a) and graph.has_node(lemma_b):
            if not graph.has_edge(edge):
                graph.add_edge(edge)
            else:
                wt = graph.edge_weight(edge)
                graph.set_edge_properties(edge, weight=wt+1)
    # print ''


def _process_first_window(graph, tokens, split_text):
    first_window = _get_first_window(split_text)
    for word_a, word_b in _combinations(first_window, 2):
        _set_graph_edge(graph, tokens, word_a, word_b)


def _init_queue(split_text):
    queue = _Queue()
    first_window = _get_first_window(split_text)
    for word in first_window[1:]:
        queue.put(word)
    return queue


def _process_word(graph, tokens, queue, word):
    for word_to_compare in _queue_iterator(queue):
        _set_graph_edge(graph, tokens, word, word_to_compare)


def _update_queue(queue, word):
    queue.get()
    queue.put(word)
    assert queue.qsize() == (WINDOW_SIZE - 1)


def _process_text(graph, tokens, split_text):
    queue = _init_queue(split_text)
    for i in xrange(WINDOW_SIZE, len(split_text)):
        word = split_text[i]
        _process_word(graph, tokens, queue, word)
        _update_queue(queue, word)


def _queue_iterator(queue):
    iterations = queue.qsize()
    for i in xrange(iterations):
        var = queue.get()
        yield var
        queue.put(var)


def _set_graph_edges(graph, tokens, split_text):
    _process_first_window(graph, tokens, split_text)
    _process_text(graph, tokens, split_text)


def _extract_tokens(lemmas, scores, ratio, words):
    lemmas.sort(key=lambda s: scores[s], reverse=True)

    # If no "words" option is selected, the number of sentences is
    # reduced by the provided ratio, else, the ratio is ignored.
    length = len(lemmas) * ratio if words is None else words
    length = int(min(length, len(lemmas)))
    return [(scores[lemmas[i]], lemmas[i],) for i in range(int(length))], length


def _lemmas_to_words(tokens):
    lemma_to_word = {}
    for word, unit in iteritems(tokens):
        lemma = unit.token
        if lemma in lemma_to_word:
            lemma_to_word[lemma].append(word)
        else:
            lemma_to_word[lemma] = [word]
    return lemma_to_word


def _get_keywords_with_score(extracted_lemmas, lemma_to_word):
    """
    :param extracted_lemmas:list of tuples
    :param lemma_to_word: dict of {lemma:list of words}
    :return: dict of {keyword:score}
    """
    keywords = {}
    for score, lemma in extracted_lemmas:
        keyword_list = lemma_to_word[lemma]
        for keyword in keyword_list:
            keywords[keyword] = score
    return keywords


def _strip_word(word):
    stripped_word_list = list(_tokenize_by_word(word))
    return stripped_word_list[0] if stripped_word_list else ""


def _get_combined_keywords(_keywords, split_text, ngram=None):
    """
    :param keywords:dict of keywords:scores
    :param split_text: list of strings
    :return: combined_keywords:list
    """
    # print split_text
    result = set()
    _punctuations = '!"#$%\'()*,.:;<=>?@[\\]^_`{|}~'
    _keywords = _keywords.copy()
    len_text = len(split_text)
    ii = 0
    for i in xrange(len_text):
        if i < ii:
            continue
        word = _strip_word(split_text[i])
        if word in _keywords:
            combined_word = [word]
            if split_text[i].strip()[-1] in _punctuations:
                if (ngram is None) or ngram == 1:
                    result.add(word) # we should not combine separated keywords
                continue
            if i + 1 == len_text:
                if (ngram is None) or ngram == 1:
                    result.add(word)   # appends last word if keyword and doesn't iterate
            for j in xrange(i + 1, len_text):
                if ngram is not None and j - i > ngram:
                    break
                other_word = _strip_word(split_text[j])
                if other_word in _keywords: #  and other_word == split_text[j]:
                    combined_word.append(other_word)
                else:
                    # for keyword in combined_word:
                    #    _keywords.pop(keyword)i
                    if ngram is None or ngram ==  len(combined_word):
                        result.add(" ".join(combined_word))
                        combined_word = []
                        ii = j + 1
                    break    
                if split_text[j].strip()[-1] in _punctuations:
                    if (ngram is None) or ngram == len(combined_word):
                        result.add(" ".join(combined_word)) # we should not combine separated keywords
                        combined_word = []
                        ii = j + 1
                        
                    break
            if combined_word and ((ngram is None) or ngram == len(combined_word)):
               result.add(" ".join(combined_word))
               combined_word = []
               ii = len_text
    return list(result)


def _get_average_score(concept, _keywords):
    word_list = concept.split()
    word_counter = 0
    total = 0
    max_score = abs(max(_keywords.values()))
    for word in word_list:
        factor = _keywords[word] / max_score
        if factor < 0.5:
            factor = -0.1 if _keywords[word] > 0 else 0.1
        elif factor > 0.999:
            factor = 1
        else:
            factor = 0.1
        total += factor * _keywords[word]
        word_counter += 1
    return total # / word_counter


def _format_results(_keywords, combined_keywords, split, scores, length):
    """
    :param keywords:dict of keywords:scores
    :param combined_keywords:list of word/s
    """
    combined_keywords.sort(key=lambda w: _get_average_score(w, _keywords), reverse=True)
    combined_keywords_ = combined_keywords[:length]
    if scores:
        return [(word, _get_average_score(word, _keywords)) for word in combined_keywords_]
    if split:
        return combined_keywords_
    return "\n".join(combined_keywords_)


def keywords(text, ratio=0.2, words=None, split=False, scores=False, pos_filter=['NN', 'JJ'], lemmatize=False, nltk_tag = False, ngram=None):
    # Gets a dict of word -> lemma
    text = to_unicode(text)
    tokens = _clean_text_by_word(text, nltk_tag)
    split_text = list(_tokenize_by_word(text, keep_punct=True))
    # print split_text
    # Creates the graph and adds the edges
    graph = _build_graph(_get_words_for_graph(tokens, pos_filter))
    split_text2 = _preprocess_strings(split_text, DEFAULT2_FILTERS)
    # print "FOR EDGE CONSTRUCTION: ", split_text2
    _set_graph_edges(graph, tokens, split_text2)
    del split_text  # It's no longer used
    # print "GRAPH NODES: ", graph.nodes()
    _remove_unreachable_nodes(graph)
    # print "STRIPPED GRAPH NODES: ", graph.nodes()
    # Ranks the tokens using the PageRank algorithm. Returns dict of lemma -> score
    pagerank_scores = _pagerank(graph)
    # print repr(pagerank_scores)
    extracted_lemmas, length = _extract_tokens(graph.nodes(), pagerank_scores, ratio, words)
    # print extracted_lemmas, length
    # The results can be polluted by many variations of the same word
    if lemmatize:
        lemmas_to_word = {}
        for word, unit in iteritems(tokens):
            if unit.token in lemmas_to_word:
                if len(word) < len(lemmas_to_word[unit.token][0]):
                    lemmas_to_word[unit.token] = [word] # try to keep the shortest word for the lemma
            else:
                lemmas_to_word[unit.token] = [word]
    else:
        lemmas_to_word = _lemmas_to_words(tokens)

    keywords = _get_keywords_with_score(extracted_lemmas, lemmas_to_word)
    # text.split() to keep numbers and punctuation marks, so separeted concepts are not combined
    combined_keywords = _get_combined_keywords(keywords, text.split(), ngram)

    return _format_results(keywords, combined_keywords, split, scores, length)


def get_graph(text):
    tokens = _clean_text_by_word(text)
    split_text = list(_tokenize_by_word(text))

    graph = _build_graph(_get_words_for_graph(tokens))
    _set_graph_edges(graph, tokens, split_text)

    return graph
