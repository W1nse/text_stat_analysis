# Количество разработчиков: 1
# Количество веток: 1
# Количество копирования: 1
# Регулярность использования: 2
# Статистический анализ текстов для курса "Автоматическая обработка текста" на ФКН НИУ ВШЭ.
## Пузырев Дмитрий Александрович

Репозиторий содержит два файла с кодом:

<b>stat_analysis.py</b>: класс с методами подсчета статистик. Включает в себя следующие функции:

1. <b>get_symbols_amount</b> — подсчет числа символов в тексте, с пробелами, без пробелов и без знаков препинания (то есть все буквенные символы)
2. <b>get_sentences_amount</b> — подсчет числа предложений в тексте, среднего числа непробельных и буквенных символов на предложение
3. <b>get_token_amount</b> — подсчет числа токенов, среднего числа токенов на предложение
4. <b>get_words_amount</b> — подсчет числа словоупотреблений, среднего числа словоупотреблений на предложение
5. <b>get_mean_word_length</b> — подсчет средней длины словоупотребления в тексте
6. <b>get_mean_word_length</b> — подсчет числа уникальных словоформ в тексте
7. <b>get_word_frequency</b> — вывод заданного числа наиболее частотных словоформ текста с частотами и отн. частотам, прорисовка круговой диаграммы частот
8. <b>get_word_pos_frequency</b> — подсчет частот частей речи по словоформам, прорисовка диаграммы (здесь и в дальнейшем лейблы граммем по стандарту opencorpora http://opencorpora.org/dict.php?act=gram)
9. <b>get_noun_cases_frequency</b>, <b>get_adj_cases_frequency</b> — подсчет частот падежей существительных и прилагательных, прорисовка диаграммы (opencorpora)
10. <b>get_noun_numbers_frequency</b>, <b>get_adj_numbers_frequency</b>, <b>get_verb_numbers_frequency</b> — подсчет частот чисел существительных, прилагательных, глаголов, прорисовка диаграммы (opencorpora)
11. <b>get_verb_pers_frequency</b> — подсчет частот лицевых форм глагола, прорисовка диаграммы (opencorpora)
12. <b>get_lemms_amount</b> — подсчет числа лемм в тексте
13. <b>get_richness</b> — коэффициент лексического богатства текста
14. <b>get_lemm_frequency</b> — вывод заданного числа наиболее частотных лемм текста с частотами и отн. частотам, прорисовка круговой диаграммы частот
15. <b>get_lemm_pos_frequency</b> — подсчет частот частей речи по леммам, прорисовка диаграммы
16. <b>get_pos_lemm_frequencies</b> — вывод заданного числа наиболее частотных лемм указанной по лейблу части речи, прорисовка диаграмм
17. <b>check_zipf_words</b>, <b>check_zipf_lemms</b> — прорисовка графика зависимости отн. частоты словоформы/леммы от ее ранга; аппроксимация указанной зависимости законом Ципфа с C=относительнольной частоте самого распространённого слова
18. <b>draw_pie</b> — служебная функция для прорисовки диаграмм
19. <b>write_log_to_path</b> — служебная функция для записи всех запрошенных статистик в указанный текстовый файл

Класс инициализируется созданием метода StatAnalysis(). Для инициализации необходим путь к текстовому файлу (при инициализации класс сам его откроет). Опционально можно поставить флаг make_log на True, тогда вся статистика будет записана в одну строку.

Пример использования отражен в тестовом скрипте <b>test.py</b>. Его можно запустить из командной строки следующим образом:

```
python test.py <путь к тексту> <True или False в зависимости от желания получить текстовый файл с результатами> --log <путь к логу опционально>
```
Все возможные статистики будут выведены на экран.

В репозитории содержаться результаты полного анализа второй главы "Алисы в стране чудес" Льюиса Кэролла и второй главы "Generation П" Виктора Пелевина.
