import re
import matplotlib.pyplot as plt
from nltk import sent_tokenize, word_tokenize, FreqDist
from matplotlib import cm, colors
import numpy as np
import pymorphy2


class StatAnalysis:
    def __init__(self, text_file, make_log=False):
        self.morph = pymorphy2.MorphAnalyzer()
        txt = open(text_file, "r")
        self.text = ''
        for line in txt:
            self.text += line
        self.text_symbols = re.sub(r'\s', '', self.text)
        self.text_letters = re.sub(r'[^А-Яа-я]', '', self.text)
        self.sentences = sent_tokenize(self.text)
        self.sent_tokens = [word_tokenize(s) for s in self.sentences]
        self.sent_words = []
        self.word_list = []
        for sent in self.sent_tokens:
            words = []
            for token in sent:
                pos_tag = self.morph.parse(token)[0][1]
                if str(pos_tag) != 'PNCT':
                    words.append(self.morph.parse(token)[0][0])
                    self.word_list.append(self.morph.parse(token)[0][0])
            self.sent_words.append(words)
        self.sent_word_lens = [len(s) for s in self.sent_words]
        self.word_set = set(self.word_list)
        self.lemm_list = []
        for word in self.word_list:
            lemm = self.morph.parse(word)[0].normal_form
            self.lemm_list.append(lemm)
        self.lemm_set = set(self.lemm_list)
        if make_log:
            self.log = ''

    def draw_pie(self, freqs, labels, title, fig_save):
        norm = colors.Normalize(vmin=0, vmax=len(freqs))
        scalarmap = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('plasma'))
        seg_colors = [scalarmap.to_rgba(i) for i in range(len(freqs))]
        plt.figure(figsize=(20, 20))
        plt.pie(freqs, labels=labels, autopct='%1.1f%%', startangle=90, colors=seg_colors, textprops={'fontsize': 20})
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        plt.title(title, fontsize=20)
        plt.savefig(fig_save)

    def get_symbols_amount(self):
        print('Общее количество символов в тексте, включая разделительные: ', len(self.text))
        self.log += 'Общее количество символов в тексте, включая разделительные: %d' % len(self.text) + '\n'
        print('Общее количество непробельных символов в тексте: %d (%.1f%%)' % (
        len(self.text_symbols), len(self.text_symbols) / len(self.text) * 100))
        self.log += 'Общее количество непробельных символов в тексте: %d (%.1f%%)' % (
        len(self.text_symbols), len(self.text_symbols) / len(self.text) * 100) +'\n'
        print('Общее количество буквенных символов в тексте: %d (%.1f%%)' % (
        len(self.text_letters), len(self.text_letters) / len(self.text) * 100))
        self.log += 'Общее количество буквенных символов в тексте: %d (%.1f%%)' % (
        len(self.text_letters), len(self.text_letters) / len(self.text) * 100) + '\n'
        self.log += '----------------------------------------------------------\n'

    def get_sentences_amount(self):
        print('Число предложений в тексте: ', len(self.sentences))
        self.log += 'Число предложений в тексте: %d' % len(self.sentences) + '\n'
        print('Среднее число непробельных символов на предложение текста: %.2f' % (
                    len(self.text_symbols) / len(self.sentences)))
        self.log += 'Среднее число непробельных символов на предложение текста: %.2f' % (
                    len(self.text_symbols) / len(self.sentences)) + '\n'
        print('Среднее число буквенных символов на предложение текста: %.2f' % (len(self.text_letters) / len(self.sentences)))
        self.log += 'Среднее число буквенных символов на предложение текста: %.2f' % (len(self.text_letters) / len(self.sentences)) + '\n'
        self.log += '----------------------------------------------------------\n'

    def get_token_amount(self):
        sent_token_lens = [len(s) for s in self.sent_tokens]
        print('Общее число токенов в тексте: ', sum(sent_token_lens))
        self.log += 'Общее число токенов в тексте: %d' % sum(sent_token_lens) +'\n'
        print('Среднее число токенов на предложение текста: %.2f' % np.mean(sent_token_lens))
        self.log += 'Среднее число токенов на предложение текста: %.2f' % np.mean(sent_token_lens) + '\n'
        self.log += '----------------------------------------------------------\n'

    def get_words_amount(self):
        print('Общее число словоупотреблений в тексте: ', sum(self.sent_word_lens))
        self.log += 'Общее число словоупотреблений в тексте: %d' % sum(self.sent_word_lens) + '\n'
        print('Среднее число словоупотреблений на предложение текста: %.2f' % np.mean(self.sent_word_lens))
        self.log += 'Среднее число словоупотреблений на предложение текста: %.2f' % np.mean(self.sent_word_lens) + '\n'
        self.log += '----------------------------------------------------------\n'

    def get_mean_word_length(self):
        print('Средняя длина слова в тексте: %.2f' % (len(self.text_letters) / sum(self.sent_word_lens)))
        self.log += 'Средняя длина слова в тексте: %.2f' % (len(self.text_letters) / sum(self.sent_word_lens)) + '\n'
        self.log += '----------------------------------------------------------\n'

    def get_unique_words_amount(self):
        print('Общее число уникальных словоформ в тексте (в %% от всех словоупотреблений): %d (%.1f%%)' % (
        len(self.word_set), (len(self.word_set) / sum(self.sent_word_lens) * 100)))
        self.log += 'Общее число уникальных словоформ в тексте (в %% от всех словоупотреблений): %d (%.1f%%)' % (
        len(self.word_set), (len(self.word_set) / sum(self.sent_word_lens) * 100)) + '\n'
        self.log += '----------------------------------------------------------\n'

    def get_word_frequency(self, amount, fig_save):
        fdist = FreqDist(self.word_list)
        print('Словофорома, частота, относительная частота (топ-%d):' % amount)
        self.log += 'Словофорома, частота, относительная частота (топ-%d):' % amount + '\n'
        words = []
        frequencies = []
        sumfreqs = 0
        num = 1
        for word, frequency in fdist.most_common(amount):
            words.append(word)
            frequencies.append(frequency)
            sumfreqs += frequency
            print('%d. %s %d %.2f' % (num, word, frequency, frequency / len(self.word_set)))
            self.log += '%d. %s %d %.2f' % (num, word, frequency, frequency / len(self.word_set)) + '\n'
            num += 1
        words.append('прочие')
        frequencies.append(len(self.word_set) - sumfreqs)
        self.draw_pie(frequencies, words, 'Относительная частота словоформ в тексте', fig_save)
        self.log += '----------------------------------------------------------\n'

    def get_word_pos_frequency(self, fig_save):
        pos_tags = {}
        for word in self.word_set:
            pos_tag = self.morph.parse(word)[0].tag.POS
            if pos_tag not in pos_tags:
                pos_tags[pos_tag] = [word]
            else:
                pos_tags[pos_tag].append(word)
        pos_tags_freqs = [len(pos_tags[pos]) for pos in pos_tags]

        print('Часть речи, её частота, её относительная частота:')
        self.log += 'Часть речи, её частота, её относительная частота:' + '\n'
        words = []
        frequencies = []
        sumfreqs = 0
        for num, tag in enumerate(pos_tags):
            if pos_tags_freqs[num] / len(self.word_set) > 0.02:
                words.append(tag)
                frequencies.append(pos_tags_freqs[num])
                sumfreqs += pos_tags_freqs[num]
            print('%s %d %.2f' % (tag, pos_tags_freqs[num], pos_tags_freqs[num] / len(self.word_set)))
            self.log += '%s %d %.2f' % (tag, pos_tags_freqs[num], pos_tags_freqs[num] / len(self.word_set)) + '\n'

        words = [x for _, x in sorted(zip(frequencies, words), reverse=True)]
        frequencies = sorted(frequencies, reverse=True)
        words.append('прочие')
        frequencies.append(len(self.word_set) - sumfreqs)
        self.draw_pie(frequencies, words, 'Относительная частота словоформ в тексте по частям речи', fig_save)
        self.log += '----------------------------------------------------------\n'

    def get_noun_cases_frequency(self, fig_save):
        noun_cases = {}
        for word in self.word_set:
            tag = self.morph.parse(word)[0].tag
            if tag.POS == 'NOUN':
                if tag.case not in noun_cases:
                    noun_cases[tag.case] = [word]
                else:
                    noun_cases[tag.case].append(word)
        cases_freqs = [len(noun_cases[case]) for case in noun_cases]

        print('Падеж существительных текста, частота падежа, относительная частота падежа:')
        self.log += 'Падеж существительных текста, частота падежа, относительная частота падежа:' + '\n'
        words = []
        frequencies = []
        sumfreqs = 0
        for num, tag in enumerate(noun_cases):
            if cases_freqs[num] / sum(cases_freqs) > 0.02:
                words.append(tag)
                frequencies.append(cases_freqs[num])
                sumfreqs += cases_freqs[num]
            print('%s %d %.2f' % (tag, cases_freqs[num], cases_freqs[num] / sum(cases_freqs)))
            self.log += '%s %d %.2f' % (tag, cases_freqs[num], cases_freqs[num] / sum(cases_freqs)) + '\n'
        words = [x for _, x in sorted(zip(frequencies, words), reverse=True)]
        frequencies = sorted(frequencies, reverse=True)
        if sum(cases_freqs) - sumfreqs > 0:
            words.append('прочие')
            frequencies.append(sum(cases_freqs) - sumfreqs)
        self.draw_pie(frequencies, words, 'Относительная частота падежей существительных в тексте', fig_save)
        self.log += '----------------------------------------------------------\n'

    def get_noun_numbers_frequency(self, fig_save):
        noun_numbers = {}
        for word in self.word_set:
            tag = self.morph.parse(word)[0].tag
            if tag.POS == 'NOUN':
                if tag.number not in noun_numbers:
                    noun_numbers[tag.number] = [word]
                else:
                    noun_numbers[tag.number].append(word)
        numbers_freqs = [len(noun_numbers[number]) for number in noun_numbers]

        print('Число существительных текста, частота числа, относительная частота числа:')
        self.log += 'Число существительных текста, частота числа, относительная частота числа:' + '\n'
        words = []
        frequencies = []
        for num, tag in enumerate(noun_numbers):
            words.append(tag)
            frequencies.append(numbers_freqs[num])
            print('%s %d %.2f' % (tag, numbers_freqs[num], numbers_freqs[num] / sum(numbers_freqs)))
            self.log += '%s %d %.2f' % (tag, numbers_freqs[num], numbers_freqs[num] / sum(numbers_freqs)) + '\n'
        words = [x for _, x in sorted(zip(frequencies, words), reverse=True)]
        frequencies = sorted(frequencies, reverse=True)
        self.draw_pie(frequencies, words, 'Относительная частота чисел существительных в тексте', fig_save)
        self.log += '----------------------------------------------------------\n'

    def get_adj_cases_frequency(self, fig_save):
        adj_cases = {}
        for word in self.word_set:
            tag = self.morph.parse(word)[0].tag
            if tag.POS == 'ADJF':
                if tag.case not in adj_cases:
                    adj_cases[tag.case] = [word]
                else:
                    adj_cases[tag.case].append(word)
        cases_freqs = [len(adj_cases[case]) for case in adj_cases]

        print('Падеж прилагательных текста, частота падежа, относительная частота падежа:')
        self.log += 'Падеж прилагательных тектса, частота падежа, относительная частота падежа:' + '\n'
        words = []
        frequencies = []
        sumfreqs = 0
        for num, tag in enumerate(adj_cases):
            if cases_freqs[num] / sum(cases_freqs) > 0.02:
                words.append(tag)
                frequencies.append(cases_freqs[num])
                sumfreqs += cases_freqs[num]
            print('%s %d %.2f' % (tag, cases_freqs[num], cases_freqs[num] / sum(cases_freqs)))
            self.log += '%s %d %.2f' % (tag, cases_freqs[num], cases_freqs[num] / sum(cases_freqs)) + '\n'

        words = [x for _, x in sorted(zip(frequencies, words), reverse=True)]
        frequencies = sorted(frequencies, reverse=True)
        if sum(cases_freqs) - sumfreqs > 0:
            words.append('прочие')
            frequencies.append(sum(cases_freqs) - sumfreqs)
        self.draw_pie(frequencies, words, 'Относительная частота падежей прилагательных в тексте', fig_save)
        self.log += '----------------------------------------------------------\n'

    def get_adj_numbers_frequency(self, fig_save):
        adj_numbers = {}
        for word in self.word_set:
            tag = self.morph.parse(word)[0].tag
            if tag.POS == 'ADJF':
                if tag.number not in adj_numbers:
                    adj_numbers[tag.number] = [word]
                else:
                    adj_numbers[tag.number].append(word)
        numbers_freqs = [len(adj_numbers[number]) for number in adj_numbers]

        print('Число прилагательных текста, частота числа, относительная частота числа:')
        self.log += 'Число прилагательных текста, частота числа, относительная частота числа:' + '\n'
        words = []
        frequencies = []
        for num, tag in enumerate(adj_numbers):
            words.append(tag)
            frequencies.append(numbers_freqs[num])
            print('%s %d %.2f' % (tag, numbers_freqs[num], numbers_freqs[num] / sum(numbers_freqs)))
            self.log += '%s %d %.2f' % (tag, numbers_freqs[num], numbers_freqs[num] / sum(numbers_freqs)) + '\n'
        words = [x for _, x in sorted(zip(frequencies, words), reverse=True)]
        frequencies = sorted(frequencies, reverse=True)
        self.draw_pie(frequencies, words, 'Относительная частота чисел прилагательных в тексте', fig_save)
        self.log += '----------------------------------------------------------\n'

    def get_verb_pers_frequency(self, fig_save):
        verb_pers = {}
        for word in self.word_set:
            tag = self.morph.parse(word)[0].tag
            if tag.POS == 'VERB':
                if tag.tense != 'past' and tag.mood == 'indc':
                    if tag.person not in verb_pers:
                        verb_pers[tag.person] = [word]
                    else:
                        verb_pers[tag.person].append(word)
        pers_freqs = [len(verb_pers[pers]) for pers in verb_pers]

        print('Лицо глаголов текста, частота лица, относительная частота лица:')
        self.log += 'Лицо глаголов текста, частота лица, относительная частота лица:' + '\n'
        words = []
        frequencies = []
        for num, tag in enumerate(verb_pers):
            words.append(tag)
            frequencies.append(pers_freqs[num])
            print('%s %d %.2f' % (tag, pers_freqs[num], pers_freqs[num] / sum(pers_freqs)))
            self.log += '%s %d %.2f' % (tag, pers_freqs[num], pers_freqs[num] / sum(pers_freqs)) + '\n'
        words = [x for _, x in sorted(zip(frequencies, words), reverse=True)]
        frequencies = sorted(frequencies, reverse=True)
        self.draw_pie(frequencies, words, 'Относительная частота лицевых форм глаголов в тексте', fig_save)
        self.log += '----------------------------------------------------------\n'

    def get_verb_numbers_frequency(self, fig_save):
        verb_numbers = {}
        for word in self.word_set:
            tag = self.morph.parse(word)[0].tag
            if tag.POS == 'VERB':
                if tag.number not in verb_numbers:
                    verb_numbers[tag.number] = [word]
                else:
                    verb_numbers[tag.number].append(word)
        numbers_freqs = [len(verb_numbers[number]) for number in verb_numbers]

        print('Число глаголов текста, частота, его относительная частота:')
        self.log += 'Число глагола, его частота, его относительная частота:' + '\n'
        words = []
        frequencies = []
        for num, tag in enumerate(verb_numbers):
            words.append(tag)
            frequencies.append(numbers_freqs[num])
            print('%s %d %.2f' % (tag, numbers_freqs[num], numbers_freqs[num] / sum(numbers_freqs)))
            self.log += '%s %d %.2f' % (tag, numbers_freqs[num], numbers_freqs[num] / sum(numbers_freqs)) + '\n'
        words = [x for _, x in sorted(zip(frequencies, words), reverse=True)]
        frequencies = sorted(frequencies, reverse=True)
        self.draw_pie(frequencies, words, 'Относительная частота чисел глаголов в тексте', fig_save)
        self.log += '----------------------------------------------------------\n'

    def get_lemms_amount(self):
        print('Число лемм в тексте: ', len(self.lemm_set))
        self.log += 'Число лемм в тексте: %d' % len(self.lemm_set) + '\n'
        self.log += '----------------------------------------------------------\n'

    def get_lemm_frequency(self, amount, fig_save):
        fdist = FreqDist(self.lemm_list)
        print('Лемма, её частота, относительная частота (топ-%d):' % amount)
        self.log += 'Лемма, её частота, относительная частота (топ-%d):' % amount + '\n'
        words = []
        frequencies = []
        sumfreqs = 0
        num = 1
        for word, frequency in fdist.most_common(amount):
            words.append(word)
            frequencies.append(frequency)
            sumfreqs += frequency
            print('%d. %s %d %.2f' % (num, word, frequency, frequency / len(self.lemm_set)))
            self.log += '%d. %s %d %.2f' % (num, word, frequency, frequency / len(self.lemm_set)) + '\n'
            num += 1
        words.append('прочие')
        frequencies.append(len(self.lemm_set) - sumfreqs)
        self.draw_pie(frequencies, words, 'Относительная частота лемм в тексте', fig_save)
        self.log += '----------------------------------------------------------\n'

    def get_richness(self):
        print('Коэффициент лексического богатства текста: %.4f' % (len(self.lemm_set) / sum(self.sent_word_lens)))
        self.log += 'Коэффициент лексического богатства текста: %.4f' % (len(self.lemm_set) / sum(self.sent_word_lens)) + '\n'
        self.log += '----------------------------------------------------------\n'

    def get_lemm_pos_frequency(self, fig_save):
        pos_tags = {}
        for word in self.lemm_set:
            pos_tag = self.morph.parse(word)[0].tag.POS
            if pos_tag not in pos_tags:
                pos_tags[pos_tag] = [word]
            else:
                pos_tags[pos_tag].append(word)
        pos_tags_freqs = [len(pos_tags[pos]) for pos in pos_tags]
        words = []
        frequencies = []
        sumfreqs = 0
        print('Часть речи, её частота, её относительная частота:')
        self.log += 'Часть речи, её частота, её относительная частота:' + '\n'
        for num, tag in enumerate(pos_tags):
            if pos_tags_freqs[num] / len(self.lemm_set) > 0.02:
                words.append(tag)
                frequencies.append(pos_tags_freqs[num])
                sumfreqs += pos_tags_freqs[num]
            print('%s %d %.2f' % (tag, pos_tags_freqs[num], pos_tags_freqs[num] / len(self.lemm_set)))
            self.log += '%s %d %.2f' % (tag, pos_tags_freqs[num], pos_tags_freqs[num] / len(self.lemm_set)) + '\n'
        words = [x for _, x in sorted(zip(frequencies, words), reverse=True)]
        frequencies = sorted(frequencies, reverse=True)
        words.append('прочие')
        frequencies.append(len(self.lemm_set) - sumfreqs)
        self.draw_pie(frequencies, words, 'Относительная частота лемм в тексте по частям речи', fig_save)
        self.log += '----------------------------------------------------------\n'

    def get_pos_lemm_frequencies(self, pos, fig_save):
        pos_list = []
        for word in self.lemm_list:
            pos_tag = self.morph.parse(word)[0].tag.POS
            if pos_tag == pos:
                pos_list.append(word)
        fdist = FreqDist(pos_list)
        print('Леммы части речи %s, их частоты, их относительные частоты (топ-10):' % pos)
        self.log += 'Леммы части речи %s, их частоты, их относительные частоты (топ-10):' % pos + '\n'
        words = []
        frequencies = []
        sumfreqs = 0
        num = 1
        for word, frequency in fdist.most_common(10):
            if frequency / len(pos_list) > 0.015:
                words.append(word)
                frequencies.append(frequency)
                sumfreqs += frequency
            print('%d. %s %d %.2f' % (num, word, frequency, frequency / len(pos_list)))
            self.log += '%d. %s %d %.2f' % (num, word, frequency, frequency / len(pos_list)) + '\n'
            num += 1
        words = [x for _, x in sorted(zip(frequencies, words), reverse=True)]
        frequencies = sorted(frequencies, reverse=True)
        words.append('прочие')
        frequencies.append(len(pos_list) - sumfreqs)
        self.draw_pie(frequencies, words, 'Относительная частота лемм части речи %s в тексте' % pos, fig_save)
        self.log += '----------------------------------------------------------\n'

    def check_zipf_words(self, fig_save):
        fdist = FreqDist(self.word_list)
        freqs = []
        for _, freq in fdist.most_common(500):
            freqs.append(freq / len(self.word_list))
        plt.figure(figsize=(20, 20))
        plt.plot(np.arange(1, 501, step=1), freqs, c='green', label='Реальность')
        zipf = [(freqs[0] / (num + 1)) for num, freq in enumerate(freqs)]
        plt.plot(np.arange(1, 501, step=1), zipf, c='red', label='Аппроксимация')
        plt.xticks(np.arange(1, 500, step=100))
        plt.grid()
        plt.xlabel('Ранг слова', fontsize=20)
        plt.ylabel('Отн. частота слова', fontsize=20)
        plt.legend(fontsize=20)
        plt.title('Проверка закона ципфа на словоформах', fontsize=20)
        plt.savefig(fig_save)

    def check_zipf_lemms(self, fig_save):
        fdist = FreqDist(self.lemm_list)
        freqs = []
        for _, freq in fdist.most_common(500):
            freqs.append(freq / len(self.lemm_list))
        plt.figure(figsize=(20, 20))
        plt.plot(np.arange(1, 501, step=1), freqs, c='green', label='Реальность')
        zipf = [(freqs[0] / (num + 1)) for num, freq in enumerate(freqs)]
        plt.plot(np.arange(1, 501, step=1), zipf, c='red', label='Аппроксимация')
        plt.xticks(np.arange(1, 500, step=100))
        plt.grid()
        plt.xlabel('Ранг слова', fontsize=20)
        plt.ylabel('Отн. частота слова', fontsize=20)
        plt.legend(fontsize=20)
        plt.title('Проверка закона ципфа на леммах', fontsize=20)
        plt.savefig(fig_save)

    def write_log_to_path(self, log_path):
        with open(log_path, "w") as log_text:
            log_text.write(self.log)
