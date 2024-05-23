# simple_reccomendation_system
Для похожих фильмов:
1)запускаешь similar_film_preprocess передав таблицу с данными о фильмах, создается файл matrix.npy
2)запускаешь similar_films передав ту же таблицу, id фильма,на основе которого ищем похожие и матрицу
3)сохраняется 'similar_results.csv'

Для рекомендованных:
1)запускаешь rating_preprocess передав таблицу рейтинга,сохранится файл algo
2)запускаешь rating_ids передав таблицу рейтинга, id пользователя и algo
3)сохраняется 'reccomend_results.csv'
