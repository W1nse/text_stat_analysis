from stat_analysis import StatAnalysis
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser(description='Statistical analysis of a given texts')
    parser.add_argument('text', help='path to .txt file for analysis')
    parser.add_argument('logging', help='option to make text log of stat analysis results (True or False)', type=bool)
    parser.add_argument('--log', help='path to log file for analysis results')
    args = parser.parse_args()

    text = StatAnalysis(args.text, args.logging)
    text.get_symbols_amount()
    text.get_sentences_amount()
    text.get_token_amount()
    text.get_words_amount()
    text.get_mean_word_length()
    text.get_unique_words_amount()
    text.get_word_frequency(10, 'text_word_frequency.jpg')
    text.get_word_pos_frequency('text_word_pos_frequency.jpg')
    text.get_noun_cases_frequency('text_noun_case_frequency.jpg')
    text.get_noun_numbers_frequency('text_noun_number_frequency.jpg')
    text.get_adj_cases_frequency('text_adj_case_frequency.jpg')
    text.get_adj_numbers_frequency('text_adj_number_frequency.jpg')
    text.get_verb_pers_frequency('text_verb_pers_frequency.jpg')
    text.get_verb_numbers_frequency('text_verb_number_frequency.jpg')
    text.get_lemms_amount()
    text.get_lemm_frequency(10, 'text_lemm_word_frequency.jpg')
    text.get_richness()
    text.get_lemm_pos_frequency('text_lemm_pos_frequency.jpg')
    text.get_pos_lemm_frequencies('NOUN', 'text_lemm_NOUN_frequency.jpg')
    text.check_zipf_words('text_zipf_word_frequency.jpg')
    text.check_zipf_lemms('text_zipf_lemm_frequency.jpg')
    text.write_log_to_path(args.log)