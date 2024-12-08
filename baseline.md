Построение бейзлайна и выбор метрик

Для построяние стартовой точке в гонке была создана Функция предсказания самого частого класса (случайный выбор с распределением весов по меткам классов в  обучающей выборке).
На наших данных функция дала следующее распределение классов:

 0    42
-1    30
 1    25
Name: count, dtype: int64
 
На всех данных (ункция предсказания самого частого класса):
Accuracy: 0.35051546391752575
Error rate: 0.6494845360824743


Функция предсказания самого частого класса на test
              precision    recall  f1-score   support

          -1       0.20      0.33      0.25         6
           0       0.44      0.36      0.40        11
           1       0.20      0.14      0.17         7

    accuracy                           0.29        24
   macro avg       0.28      0.28      0.27        24
weighted avg       0.31      0.29      0.29        24


+
Далее начали гонку.
## Bag of words + LogReg

LogisticRegression() на мешке слов
              precision    recall  f1-score   support

          -1       0.67      0.67      0.67         6
           0       0.64      0.64      0.64        11
           1       0.71      0.71      0.71         7

    accuracy                           0.67        24
   macro avg       0.67      0.67      0.67        24
weighted avg       0.67      0.67      0.67        24



+

## TF-IDF + LogReg
LogisticRegression() на TF-IDF
              precision    recall  f1-score   support

          -1       0.67      0.67      0.67         6
           0       0.64      0.64      0.64        11
           1       0.71      0.71      0.71         7

    accuracy                           0.67        24
   macro avg       0.67      0.67      0.67        24
weighted avg       0.67      0.67      0.67        24

+

## N-GRAM BAG OF WORDS + LogReg

LogisticRegression() на N-GRAM BAG OF WORDS
              precision    recall  f1-score   support

          -1       1.00      0.62      0.77         8
           0       0.59      1.00      0.74        10
           1       1.00      0.43      0.60         7

    accuracy                           0.72        25
   macro avg       0.86      0.68      0.70        25
weighted avg       0.84      0.72      0.71        25
+
## W2V + LogReg

Здесь качество упало

LogisticRegression() на word to vec
              precision    recall  f1-score   support

          -1       0.00      0.00      0.00         8
           0       0.40      1.00      0.57        10
           1       0.00      0.00      0.00         7

    accuracy                           0.40        25
   macro avg       0.13      0.33      0.19        25
weighted avg       0.16      0.40      0.23        25

+
Создали пайплайн на TF-IDF + LogReg хотя качество на N-GRAM BAG OF WORDS + LogReg получили выше.

В качестве метрик качество использовали classification_report():
1)  accuracy 
2) precision 
3) recall
4) f1-score


Таким образом, использование этих метрик позволяет провести детальный анализ эффективности различных подходов к решению задачи классификации.
