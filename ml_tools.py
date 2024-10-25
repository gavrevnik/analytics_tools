import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, cv, Pool
from sklearn.metrics import roc_auc_score, classification_report


### CatBoostClassifier
def init_catboostclassifier(params = None, train_pool = None, test_pool = None):
    """Инициализирует модель обучения с заданными параметрами params
    train, test_pool - датасеты с тренировочными и тестовыми данными модели
    """
    if params is None:
        params = {
                'depth': 6, # глубина деревьев (больше = сложнее зависимости, но дольше и риск переобуч)
                'learning_rate': 0.03, # шаг изменения параметров при добавлении нового дерева в ансамбль (аналогично depth)
                'iterations': 1000, # количество деревьев в ансамбле
                'l2_leaf_reg': 3, # штраф за большие веса в листьях (аналогично рег. в регрессии)
                'bagging_temperature': 1, # интенсивность рандомизации при бустрапе в ансамбле
                'auto_class_weights': 'Balanced', # автоматически присвоить веса меткам обр. пропорционально частоте
                'random_strength': 1, # рандомизация сплитования (также влияет на переобучение)
                'loss_function' : 'Logloss', # функция потерь которую оптимизиуем при обучении дерева
                'eval_metric' : 'F1', # метрика, которую оптимизирует ансамбль
                'verbose' : False, # не выводить логи с каждой иттерации (можно указать частоту)
            #     'silent' : False # полностью отключить вывод логов модели когда true (задается либо silent либо verbose)
                }
    model = CatBoostClassifier(**params)
    result = None
    if train_pool is not None:
        model.fit(train_pool, eval_set=test_pool, use_best_model=True)
        predict_vals = model.predict(test_pool)
        predict_proba_vals = model.predict_proba(test_pool)[:, 1] # 0 - предикт метки 0, 1 = 1, ...
        df_importance = pd.DataFrame({'feature' : train_pool.get_feature_names(),
                      'feature_importance' : model.feature_importances_}).sort_values(by='feature_importance', ascending=False)
        result = {'predict_vals' : predict_vals, 'predict_proba_vals' : predict_proba_vals, 'df_importance' : df_importance}
    return model, result

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
    fold_count = 3-10; чем выше - тем больше точность модели, но дольше обучение
    return: оцениваемые значения для eval_metric
    """

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
