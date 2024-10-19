import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, cv, Pool
from sklearn.metrics import roc_auc_score, classification_report


### CatBoostClassifier
def get_pool(df_x, y, cat_features=None, fillna_val='NULL'):
    """Возвращаем тренировочный объект Pool для работы моделей
    df_x, y = фичи и таргет для тренировки
    cat_features = arr, список категориальных колонок
    fillna = чем заливать пропуски в категориальных фичах
    """
    if cat_features:
        df_x.loc[:, cat_features] = df_x.loc[:, cat_features].fillna(fillna_val)
        cat_features = [df_x.columns.get_loc(j) for j in cat_features]
    return Pool(data=df_x, label=y, cat_features=cat_features)

def get_from_pool(pool):
    """извлекаем обратно данные из pool в формате df_x, y
    Работает только когда в Pool нет категориальных фичей!
    """
    try:
        df_x = pd.DataFrame(pool.get_features(), columns=pool.get_feature_names())
    except:
        df_x = 'невозможно извлечь df из pool с категориальными фичами'
    y = pool.get_label()
    return df_x, y

def get_cv(params, pool, fold_count):
    """Кросс-валидация результатов модели на данных pool
    с гиперпараметрами params
    fold_count = 3-10; чем выше - тем больше точность модели, но дольше обучение"""

    cv_results = cv(
        params=params, # гиперпараметры модели которые тестим
        pool=pool,
        logging_level='Silent',
        fold_count=fold_count,  # Количество фолдов для кроссвалидации
        shuffle=True,  # Перемешивание данных
        stratified=True,  # Сохранение распределения классов
        )
    return cv_results[-1:]

### метрики качества
def get_model_score(y_test, y_predict, y_predict_probability=None):
    """
    y_test - реальные значения на тестовом сете
    y_predict - предсказания модели
    y_predict_probability - вероятность принадлежности к классу 1 (если модель дает)
    """
    report = classification_report(y_test, y_predict, digits=4, output_dict=True)
    ans = report['1']
    ans['accuracy'] = report['accuracy']
    if y_predict_probability is not None:
        ans['roc_auc'] = roc_auc_score(y_test, y_predict_probability)
    return ans
