Проект по классификации пресс-релизов Центрального банка России с прогнозным моделированием будущей ключевой ставки с использованием методов машинного обучения

План работы над проектом «Прогнозирование ключевой ставки на основе пресс-релизов ЦБ»

Разведочный анализ данных и первичная аналитика
(примерный дедлайн: 15 ноября)

Задачи:

1) Сбор данных: пресс-релизы ЦБ с 2014 года, данные о других экономических показателях, которые могут влиять на ключевую ставку (инфляция, ВВП, курс валюты, цена барреля нефти). Занимаются Лия Сердюк и Евгений Крылов.
2) Предобработка данных, разделение данных на обучающую, тестовую и валидационную выборки. Занимаются Иван Бойцов и Денис.
3) Анализ данных: визуальный анализ данных (графики, диаграммы). Занимаются Лия Сердюк и Евгений Крылов

Модели машинного обучения (ML) Задачи:

Выбор алгоритмов: протестировать катбуст, классические ML-алгоритмы, градиентный бустинг и т.д. Выбрать модель с наибольшей точностью. Обучение моделей, настройка гиперпараметров. Оценка моделей: оценка точности предсказаний моделей на тестовой выборке, анализ различных метрик, выбор лучшей модели. 

Глубокое обучение (DL) Задачи:

Подготовка данных: преобразование данных в числовые векторы, создание датасета. Выбор моделей глубокого обучения. Обучение и оценка модели. Задачи, которые хочется сделать, но, возможно, не успеем:

Анализ тональности пресс-релизов ЦБ (позитивный, отрицательный, нейтральный). Анализ чувствительности: определить чувствительность модели к изменениям в тексте пресс-релиза, чтобы понять, как различные фразы и слова влияют на прогноз.
