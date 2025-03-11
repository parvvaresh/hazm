from hazm import Stemmer, Lemmatizer, Normalizer, word_tokenize
import numpy as np
import math


def counter_element(temp):
    """این تابع عناصر موجود در لیست را شمارش می‌کند.
    
    مثال:
        >>> temp = [1, 2, 3, 2, 1, 1, 1]
        >>> counter_element(temp)
        {1: 4, 2: 2, 3: 1}
    """
    counter = {}
    for element in temp:
        counter[element] = counter[element] + 1 if element in counter else 1
    return counter


def ngrams(text, n, ignore_split=False):
    """این تابع n-gramهای متن را تولید می‌کند.
    
    مثالها:
        >>> ngrams("alireza", 3, True)
        [('a', 'l', 'i'), ('l', 'i', 'r'), ('i', 'r', 'e'), ('r', 'e', 'z'), ('e', 'z', 'a')]

        >>> ngrams("این یک متن نمونه است", 2)
        [('این', 'یک'), ('یک', 'متن'), ('متن', 'نمونه'), ('نمونه', 'است')]
    """
    if ignore_split:
        return [tuple(text[index: index + n]) for index in range(0, len(text) - n + 1)]
    else:
        words = word_tokenize(text)
        return [tuple(words[index: index + n]) for index in range(0, len(words) - n + 1)]


def overlaps_dict(dict1, dict2):
    """اشتراکات دو دیکشنری را محاسبه می‌کند.
    
    مثال:
        >>> dict1 = {"a": 3, "b": 2, "c": 7}
        >>> dict2 = {"a": 1, "b": 5, "f": 17}
        >>> overlaps_dict(dict1, dict2)
        {'a': 1, 'b': 2}
    """
    overlaps = dict()
    for key in dict1:
        if key in dict2:
            _val_overlaps = min(dict1[key], dict2[key])
            overlaps.update({key: _val_overlaps})
    return overlaps


class bleu:
    """این کلاس برای محاسبه امتیاز BLEU استفاده می‌شود.
    
    مثال:
        >>> model = bleu()
        >>> pred = "سلام من امروز نمیتوانم بیام"
        >>> refs = ["سلام نمیتونم بیام امروز", "سلام من امروز نمتونم بیایم"]
        >>> model.bleu_score(refs, pred)
        0.00010745699318235427
    """

    def __init__(self, min_n_gram: int = 1, max_n_gram: int = 4) -> None:
        self.min_n_gram = min_n_gram
        self.max_n_gram = max_n_gram
        self.normalizer = Normalizer()

    def bleu_score(self, refs: list, pred: str) -> float:
        """امتیاز BLEU را محاسبه می‌کند.
        
        Args:
            refs: لیستی از جملات مرجع
            pred: جمله پیش‌بینی شده
            
        Returns:
            امتیاز BLEU بین 0 و 1
        """
        pred = self.normalizer.normalize(pred)
        refs = [self.normalizer.normalize(ref) for ref in refs]

        pred_ngram_count = [counter_element(ngrams(pred, n)) for n in range(self.min_n_gram, self.max_n_gram + 1)]
        pred_lenght = len(word_tokenize(pred))

        refs_lenght = [len(word_tokenize(ref)) for ref in refs]
        refs_ngram_count = list()
        for n in range(self.min_n_gram, self.max_n_gram + 1):
            result = [counter_element(ngrams(ref, n)) for ref in refs]
            refs_ngram_count.append(result)

        clipped_precision = list()
        for n in range(0, self.max_n_gram):
            clipped_count = self._clipped_count(refs_ngram_count[n], pred_ngram_count[n])
            clipped_precision.append(clipped_count / sum(pred_ngram_count[n].values()))

        db = 0
        for cp in clipped_precision:
            if cp == 0:
                db += math.log(1e-15) * 0.25
            else:
                db += math.log(cp) * 0.25

        return math.exp(db) * self._brevity_penalty(self._get_closest_ref_lenght(refs_lenght, pred_lenght), pred_lenght)

    def _clipped_count(self, refs_ngram, pred_ngram):
        score = 0
        for p_gram in pred_ngram:
            pred_ngram_count = pred_ngram[p_gram]
            max_gram_refs = 0
            for ref_ngram in refs_ngram:
                if p_gram in ref_ngram:
                    max_gram_refs = max(max_gram_refs, ref_ngram[p_gram])
            score += min(max_gram_refs, pred_ngram_count)
        return score

    def _brevity_penalty(self, ref_lenght, pred_lenght):
        if pred_lenght > ref_lenght:
            return 1
        else:
            return math.exp(1 - ref_lenght / pred_lenght)

    def _get_closest_ref_lenght(self, refs_lenght, pred_lenght):
        return min(refs_lenght, key=lambda ref_lenght: abs(ref_lenght - pred_lenght))


class chrf:
    """این کلاس برای محاسبه امتیاز CHRF استفاده می‌شود.
    
    مثال:
        >>> model = chrf()
        >>> ref = "این یک توپ آبی است"
        >>> pred = "این اس یک توپ آبی"
        >>> model.chrf_score(ref, pred)
        0.7657370407370406
    """
    def __init__(self, min_size_ngram=1, max_size_ngram=6):
        self.min_size_ngram = min_size_ngram
        self.max_size_ngram = max_size_ngram
        self.beta = 3
        self.normalizer = Normalizer()

    def chrf_score(self, ref, pred):
        pred = self.normalizer.normalize(pred)
        ref = self.normalizer.normalize(ref)

        chrf_scores = []
        for n in range(self.min_size_ngram, self.max_size_ngram + 1):
            ref_ngrams = ngrams(ref, n, True)
            pred_ngrams = ngrams(pred, n, True)

            precision_recall = self._precision_recall(ref_ngrams, pred_ngrams)
            try:
                chrf = ((1 + self.beta ** 2) * precision_recall["precision"] * precision_recall["recall"]) / (self.beta ** 2 * precision_recall["precision"] + precision_recall["recall"])
            except ZeroDivisionError:
                chrf = 0
            chrf_scores.append(chrf)

        return sum(chrf_scores) / len(chrf_scores)

    def _precision_recall(self, ref_ngram, predict_ngram):
        ref_ngram_count = counter_element(ref_ngram)
        predict_ngram_count = counter_element(predict_ngram)

        tp = sum(overlaps_dict(ref_ngram_count, predict_ngram_count).values())
        tpfp = sum(predict_ngram_count.values())
        tpfn = sum(ref_ngram_count.values())
        precision = tp / tpfp if tpfp > 0 else 0
        recall = tp / tpfn if tpfn > 0 else 0

        return {
            "precision": precision,
            "recall": recall
        }


class gleu:
    """این کلاس برای محاسبه امتیاز GLEU استفاده می‌شود.
    
    مثال:
        >>> model = gleu()
        >>> pred = "سلام من امروز نمیتوانم بیام"
        >>> refs = ["سلام نمیتونم بیام امروز", "سلام من امروز نمتونم بیایم"]
        >>> model.gleu_score(refs, pred)
        0.42857142857142855
    """
    def __init__(self, min_n_gram=1, max_n_gram=4):
        self.min_n_gram = min_n_gram
        self.max_n_gram = max_n_gram
        self.normalizer = Normalizer()

    def gleu_score(self, refs, pred):
        pred = self.normalizer.normalize(pred)
        refs = [self.normalizer.normalize(ref) for ref in refs]

        pred_all_gram = self._all_gram(pred)
        tp_fp = sum(counter_element(pred_all_gram).values())

        scores = []
        for ref in refs:
            ref_all_gram = self._all_gram(ref)
            tp_fn = sum(counter_element(ref_all_gram).values())
            tp = sum(overlaps_dict(counter_element(ref_all_gram), counter_element(pred_all_gram)).values())
            precision = tp / tp_fp if tp_fp > 0 else 0
            recall = tp / tp_fn if tp_fn > 0 else 0
            scores.append(min(precision, recall))
        return max(scores)

    def _all_gram(self, text, is_list=False):
        result = []
        for n in range(self.min_n_gram, self.max_n_gram + 1):
            result.extend(ngrams(text, n))
        return result


class Meteor:
    """این کلاس برای محاسبه امتیاز METEOR استفاده می‌شود.
    
    مثال:
        >>> model = Meteor()
        >>> pred = "سلام من امروز نمیتوانم بیام"
        >>> refs = ["سلام نمیتونم بیام امروز", "سلام من امروز نمتونم بیایم"]
        >>> model.meteor_score(refs, pred)
        0.7500000000000001
    """
    def __init__(self):
        self.stemmer = Stemmer()
        self.lemmatizer = Lemmatizer()
        self.normalizer = Normalizer()

    def meteor_score(self, refs, pred):
        return max([self._meteor_single_refs_pred(ref, pred) for ref in refs])

    def _meteor_single_refs_pred(self, ref, pred):
        ref = self.normalizer.normalize(ref)
        pred = self.normalizer.normalize(pred)

        ref_num_word = self._create_num(ref)
        pred_num_word = self._create_num(pred)

        matching_word_by_word, ref_num_word, pred_num_word = self._matching_word_by_word(ref_num_word, pred_num_word)
        stemmer_matching_word_by_word, ref_num_word, pred_num_word = self._stemmer_matching_word_by_word(ref_num_word, pred_num_word)
        synonym_matching_word_by_word, ref_num_word, pred_num_word = self._synonym_matching_word_by_word(ref_num_word, pred_num_word)

        all_matches = sorted(matching_word_by_word + stemmer_matching_word_by_word + synonym_matching_word_by_word, key=lambda element: element[0])

        ref_length = len(word_tokenize(ref))
        pred_length = len(word_tokenize(pred))
        matches_length = len(all_matches)

        try:
            precision = matches_length / pred_length
            recall = matches_length / ref_length
            fscore = (precision * recall * 10) / ((9 * precision) + recall)
            chunk_count = self._chunks(all_matches)
            frag_frac = chunk_count / matches_length
        except ZeroDivisionError:
            return 0

        penalty = 0.5 * (frag_frac ** 3)
        return fscore * (1 - penalty)

    def _matching_word_by_word(self, ref_num_word, pred_num_word):
        matching_word_by_word = []
        for index_pred in range(len(pred_num_word) - 1, -1, -1):
            for index_ref in range(len(ref_num_word) - 1, -1, -1):
                if pred_num_word[index_pred][1] == ref_num_word[index_ref][1]:
                    matching_word_by_word.append((pred_num_word[index_pred][0], ref_num_word[index_ref][0]))
                    ref_num_word.pop(index_ref)
                    pred_num_word.pop(index_pred)
                    break
        return matching_word_by_word, ref_num_word, pred_num_word

    def _stemmer_matching_word_by_word(self, ref_num_word, pred_num_word):
        ref_num_word = [(index, self.stemmer.stem(word)) for index, word in ref_num_word]
        pred_num_word = [(index, self.stemmer.stem(word)) for index, word in pred_num_word]
        return self._matching_word_by_word(ref_num_word, pred_num_word)

    def _synonym_matching_word_by_word(self, ref_num_word, pred_num_word):
        matching_word_by_word = []
        for index_pred in range(len(pred_num_word) - 1, -1, -1):
            pred_lemma = self.lemmatizer.lemmatize(pred_num_word[index_pred][1])

            for index_ref in range(len(ref_num_word) - 1, -1, -1):
                ref_lemma = self.lemmatizer.lemmatize(ref_num_word[index_ref][1])

                if ref_lemma == pred_lemma:
                    matching_word_by_word.append((pred_num_word[index_pred][0], ref_num_word[index_ref][0]))
                    ref_num_word.pop(index_ref)
                    pred_num_word.pop(index_pred)
                    break
        return matching_word_by_word, ref_num_word, pred_num_word

    def _chunks(self, matches):
        if not matches:
            return 0

        index = 0
        chunk = 1

        while index < len(matches) - 1:
            if (matches[index + 1][0] == matches[index][0] + 1) and (matches[index + 1][1] == matches[index][1] + 1):
                index += 1
                continue

            chunk += 1
            index += 1
        return chunk

    def _create_num(self, text):
        words = word_tokenize(text)
        return [(index, words[index]) for index in range(len(words))]




class nist:
    """این کلاس برای محاسبه امتیاز NIST استفاده می‌شود.

        مثال:
            >>> model = nist()
            >>> pred = "سلام من امروز نمیتوانم بیام"
            >>> refs = ["سلام نمیتونم بیام امروز", "سلام من امروز نمتونم بیایم"]
            >>> model.nist_score(refs, pred)
            1.7519550008653877
    """
    def __init__(self, n=5):
        self.number_n_grams = n
        self.normalizer = Normalizer()


    def nist_score(self, refs, pred):
        pred = self.normalizer.normalize(pred)
        refs = [self.normalizer.normalize(ref) for ref in refs]

        pred_ngrams = [ngrams(pred, n) for n in range(1, self.number_n_grams + 1)]
        refs_ngrams = [[ngrams(ref, n) for n in range(1, self.number_n_grams + 1)] for ref in refs]

        refs_ngrams_freq, refs_total_words = dict(), 0
        for index in range(len(refs)):
            refs_total_words += len(refs[index].split())
            for num_gram in range(self.number_n_grams):
                count_ngrams = counter_element(refs_ngrams[index][num_gram])
                for element in count_ngrams:
                    refs_ngrams_freq[element] = (
                        refs_ngrams_freq[element] + count_ngrams[element]
                        if element in refs_ngrams_freq else 1
                    )

        info = self._info(refs_ngrams_freq, refs_total_words)

        nist_score, pred_length, ref_length = [], len(pred.split()), 0
        for n in range(self.number_n_grams):
            nist_precisions = []
            for index_ref in range(len(refs)):
                ref_count_gram = counter_element(refs_ngrams[index_ref][n])
                pred_count_gram = counter_element(pred_ngrams[n])
                overlaps_ngram = overlaps_dict(ref_count_gram, pred_count_gram)
                numerator = sum(info[n_gram] * count for n_gram, count in overlaps_ngram.items())
                denominator = sum(pred_count_gram.values())
                nist_precisions.append(numerator / denominator)

            nist_score.append(max(nist_precisions))
            ref_length += len(refs[nist_precisions.index(max(nist_precisions))].split())

        return sum(nist_score) * self._length_penalty((ref_length / self.number_n_grams), pred_length)

    def _info(self, refs_ngrams_freq, refs_total_words):
        info = {}
        for grams_1n in refs_ngrams_freq:
            grams_1m = grams_1n[:-1]  # w1, w2, ... wn-1
            occurrence = refs_ngrams_freq.get(grams_1m, refs_total_words)
            info[grams_1n] = math.log(occurrence / refs_ngrams_freq[grams_1n], 2)
        return info

    def _length_penalty(self, refs_length, pred_length):
        ratio = pred_length / refs_length
        if 0 < ratio < 1:
            beta = math.log(0.5) / (math.log(1.5) ** 2)
            return math.exp(beta * (math.log(ratio) ** 2))
        return 1





class wer:
    """این کلاس برای محاسبه نرخ خطای کلمه (WER) استفاده می‌شود.
    
    مثال:
        >>> model = wer()
        >>> ref = "سگ زیر میز است"
        >>> pred = "سگ زیر نیز هست"
        >>> model.wer_score(ref, pred)
        0.5
        >>> model.get_detail()
        {'delete': 0, 'insert': 1, 'substitution': 2, 'same words': 3}
    """
    def __init__(self):
        self.normalizer = Normalizer()

    def wer_score(self, ref, pred):
        self.pred = self._preprocess(pred)
        self.ref = self._preprocess(ref)

        costs = np.zeros((1 + len(self.pred), 1 + len(self.ref)))
        self.backtrace = np.zeros((1 + len(self.pred), 1 + len(self.ref)))

        costs[0] = [j for j in range(0, len(self.ref) + 1)]
        self.backtrace[0][:] = 2

        costs[:, 0] = [j for j in range(0, len(self.pred) + 1)]
        self.backtrace[:, 0] = 3

        self.backtrace[0, 0] = 10  

        for row in range(1, len(self.pred) + 1):
            for col in range(1, len(self.ref) + 1):
                if self.pred[row - 1] == self.ref[col - 1]:
                    costs[row, col] = costs[row - 1, col - 1]
                    self.backtrace[row, col] = 1
                else:
                    substitution = costs[row - 1, col - 1]
                    delete = costs[row - 1, col]
                    insert = costs[row, col - 1]
                    final_cost = min(delete, insert, substitution)
                    costs[row, col] = final_cost + 1

                    if final_cost == delete:
                        self.backtrace[row, col] = 3
                    elif final_cost == insert:
                        self.backtrace[row, col] = 2
                    elif final_cost == substitution:
                        self.backtrace[row, col] = 4

        return costs[-1, -1] / len(self.ref)

    def _preprocess(self, text):

        text = self.normalizer.normalize(text)
        return word_tokenize(text)

    def get_detail(self):
        i, j = len(self.pred), len(self.ref)
        num_same, num_del, num_sub, num_ins = 0, 0, 0, 0

        while i > 0 or j > 0:
            if self.backtrace[i, j] == 1:
                i -= 1
                j -= 1
                num_same += 1
            elif self.backtrace[i, j] == 4:
                i -= 1
                j -= 1
                num_sub += 1
            elif self.backtrace[i, j] == 2:
                j -= 1
                num_ins += 1
            elif self.backtrace[i, j] == 3:
                i -= 1
                num_del += 1

        return {
            "delete": num_del,
            "insert": num_ins,
            "substitution": num_sub,
            "same words": num_same,
        }



