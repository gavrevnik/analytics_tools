import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, cv, Pool
from sklearn.metrics import roc_auc_score, classification_report, mean_absolute_error, mean_squared_error,r2_score, mean_absolute_percentage_error
from  sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import learning_curve, validation_curve, cross_validate, RandomizedSearchCV, GridSearchCV
import matplotlib.pyplot as plt


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

def get_catboost_cv(params, pool, fold_count):
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

### МЕТРИКИ КАЧЕСТВА
def get_model_score(y_test, y_predict, is_clf=True, y_predict_probability=None):
    """
    is_clf - является ли модель классификатором. Если нет - то оцениваем как регрессию
    y_test - реальные значения на тестовом сете
    y_predict - предсказания модели
    y_predict_probability - вероятность принадлежности к классу 1 (если модель дает)
    """
    if is_clf:
        report = classification_report(y_test, y_predict, digits=4, output_dict=True)
        ans = report['1']
        ans['accuracy'] = report['accuracy']
        if y_predict_probability is not None:
            ans['roc_auc'] = roc_auc_score(y_test, y_predict_probability)
    else:
        mae = mean_absolute_error(y_test, y_predict)
        rmse = np.sqrt(mean_squared_error(y_test, y_predict))
        r2 = r2_score(y_test, y_predict)
        mape = mean_absolute_percentage_error(y_test, y_predict)
        ans = {'R^2' : r2, 'MAE-Mean_Abs_Err' : mae, 'RMSE-Root_Mean_Sq_Err' : rmse, 'MAPE-Mean_abs_perc_err_%' : 100 * mape}
    return ans

### ПРЕПРОЦЕССИНГ
def ordinal_encoder(df, ordinal_dict, fillna_val = -1):
    """
    df = датасет с фичами; ordinal_dict = {feature_name : [f1, f2, ...], ...}
    Осуществляет кодирование порядковых фичей целыми 0,1,2.. согласно порядку в ordinal_dict
    Наличие пропусков также можно учесть в ordinal_dict
    """
    df_ = df.copy()
    categorical_features = list(ordinal_dict.keys())
    encoder = OrdinalEncoder(categories=list(ordinal_dict.values()))
    df_[categorical_features] = encoder.fit_transform(df_[categorical_features].values)
    return df_


###  ВАЛИДАЦИЯ
def get_cv(model, X, y, cv = 5):
    """Кросс валидация модели model по группе метрик качества; выводим значения avg(scoring) +- std(scoring); cv=число фолдов разбиения"""
    result_ = cross_validate(model, X, y, scoring = ['f1', 'roc_auc', 'accuracy', 'recall', 'precision'], cv = cv)
    result = {}
    for j in result_.keys():
        if j not in ['fit_time', 'score_time']:
            result[j.split('test_')[1]] = f"""{np.round(np.mean(result_[j]), 3)} +- {np.round(np.std(result_[j]), 3)}"""
    return result

def hyper_params_search(model, X, y, param_grid, scoring='f1', cv=5, n_iter = None, return_best_model = False):
    """Перебор гиперпараметров model на X, y через кросс-валидацию cv
    n_iter - число случайных семплов сетки; n_iter = None -> стандартный grid_search
    param_grid = {a : [1,2,3], ...} или вероятностно (для rand_search) param_grid = {a : uniform(loc=0, scale=4)}
    """
    if n_iter is not None:
        search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, scoring=scoring, cv=cv)
    else:
        search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv)
    search = search.fit(X, y)
    result = {'best_params' : search.best_params_,
              'best_score' : search.best_score_}
    if return_best_model:
        result['best_model'] = search.best_estimator_
    return result

def get_learning_and_validation_curve(model, X, y, scoring='f1',
                       train_sizes=None,
                       param_name=None,
                       param_range=None,
                       cv=5,
                       plot=True,
                       figsize=(15, 4)
                       ):
    """
    Строим зависимость качества обучения на тренировке/тесте от размера выборки или гиперпараметров
    train_sizes = [0.25, 0.5, 1] -> процент тренировочной выборки - исходная = len(X) * (cv-1)/cv
    param_range = [1, 2, 3] -> значения перебираемого гиперпараметра модели param_name
    plot = True - визуализация кривых обучения и валидации на месте
    return: result = список посчитанных значений
    """
    result = {}
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # обучающая кривая
    if train_sizes is not None:
        train_sizes_abs, train_scores, test_scores = learning_curve(model, X, y,
                                                                          train_sizes=train_sizes,
                                                                          scoring=scoring,
                                                                          cv=cv)


        result['learning_train_sizes'] = train_sizes_abs
        result['learning_train_scores'] = train_scores
        result['learning_test_scores'] = test_scores

        if plot == True:
            # plt.figure(figsize=figsize)  # Установка размера графика
            axs[0].plot(train_sizes_abs, train_scores.mean(axis=1), label='train_scores', color='blue', marker='o')
            axs[0].plot(train_sizes_abs, test_scores.mean(axis=1), label='test_scores', color='orange', marker='x')
            axs[0].set_title(f'learning curve - качество обучения от объема данных')
            axs[0].set_xlabel('train_sizes frac')
            axs[0].set_ylabel(f'AVG {scoring}')
            axs[0].legend()
            axs[0].grid()


    # валидационная кривая
    if param_name is not None:
        train_scores, test_scores = validation_curve(model, X, y,
                                                     param_name=param_name,
                                                     param_range=param_range,
                                                     scoring=scoring, cv=cv)
        result['validation_train_scores'] = train_scores
        result['validation_test_scores'] = test_scores

        if plot == True:
            # plt.figure(figsize=figsize)  # Установка размера графика
            axs[1].plot(param_range, train_scores.mean(axis=1), label='train_scores', color='blue', marker='o')
            axs[1].plot(param_range, test_scores.mean(axis=1), label='test_scores', color='orange', marker='x')
            axs[1].set_title(f'validation curve - качество обучения от {param_name}')
            axs[1].set_xlabel(param_name)
            axs[1].set_ylabel(f'AVG {scoring}')
            axs[1].legend()
            axs[1].grid()

    return result
