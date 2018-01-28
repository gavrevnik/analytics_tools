import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, cv, Pool
from sklearn.metrics import roc_auc_score, classification_report, mean_absolute_error, mean_squared_error,r2_score, mean_absolute_percentage_error, silhouette_score, calinski_harabasz_score
from  sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import learning_curve, validation_curve, cross_validate, RandomizedSearchCV, GridSearchCV
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import shap
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import PartialDependenceDisplay as pdp
from scipy.stats import f_oneway

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

def get_univariate_target_relation(dfx, Y, y_type = 'd', anova_calc = True, discrete_features='auto', random_state = 42, alpha=0.05):
    """Получаем связь фичей с таргетом - корреляции, MI
    dfx - датасет с признаками; Y - таргет; y_type = d (discrete, clf), c (continuous, regr)
    discrete_features = список фичей (arr) либо авто-определение по bool типам
    anova_calc = проверка значимости фичей относительно таргета (только для y_type = d)
    return df = виды корреляций + MI взаимосвязи + anova_check"""
    dfx = dfx.copy()
    # mutual information
    mi_model = [mutual_info_classif if y_type == 'd' else mutual_info_regression][0]
    mi_vals = mi_model(dfx, Y, discrete_features = discrete_features, random_state = random_state)
    df = pd.DataFrame({'features' : dfx.columns, 'mi': mi_vals})
    # correlation
    dfx['target_y'] = Y
    df['corr_pearson'] = dfx.corr(method='pearson')['target_y'].values[:-1]
    df['corr_spearman_rang'] = dfx.corr(method='spearman')['target_y'].values[:-1]
    # anova = проверка что значения таргета зависят от среднего признаков
    if (anova_calc == True) and (y_type == 'd'):
        results = []
        for col in dfx.drop(columns = ['target_y']).columns:
            groups = [dfx[col][Y == category] for category in np.unique(Y)]
            _, p_value = f_oneway(*groups)
            results.append(p_value)
        df['anova_pval'] = results
        df['anova_is_feature_useful'] = (df.anova_pval < alpha).astype(int)
    return df

###  ВАЛИДАЦИЯ
def get_cv(model, X, y, cv = 5, is_clf=1):
    """Кросс валидация модели model по группе метрик качества
    cv=число фолдов разбиения
    is_clf: 0 = регрессия, 1 = классификация
    return ср. метрики качества по фолдам для тренировки и теста
    """
    if is_clf == 1:
        scoring_list = ['f1', 'f1_macro', 'f1_micro', 'roc_auc', 'accuracy', 'recall', 'precision']
    else:
        scoring_list = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error']
    result_ = cross_validate(model, X, y, scoring = scoring_list, cv = cv, return_train_score=True)
    test_dct, train_dct = {}, {}
    for j in [j for j in result_.keys() if 'test_' in j]:
        test_dct[j.split('test_')[1]] = f"""{np.round(np.mean(result_[j]), 4)} +- {np.round(np.std(result_[j]), 4)}"""
    for j in [j for j in result_.keys() if 'train_' in j]:
        train_dct[j.split('train_')[1]] = f"""{np.round(np.mean(result_[j]), 4)} +- {np.round(np.std(result_[j]), 4)}"""
    df_test = pd.DataFrame({'metric' : list(test_dct.keys()), 'test_score' : list(test_dct.values())})
    df_train = pd.DataFrame({'metric' : list(train_dct.keys()), 'train_score' : list(train_dct.values())})
    return df_test.merge(df_train)

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
                       train_sizes=None, param_name=None, param_range=None,
                       cv=5, plot=True, figsize=(15, 4)):
    """
    Строим зависимость качества обучения на тренировке/тесте от размера выборки или гиперпараметров
    train_sizes = [0.25, 0.5, 1] -> процент тренировочной выборки - исходная = len(X) * (cv-1)/cv
    param_range = [1, 2, 3] -> значения перебираемого гиперпараметра модели param_name
    plot = True - визуализация кривых обучения и валидации на месте
    return: result = список посчитанных значений
    """
    result = {}
    _, axs = plt.subplots(1, 2, figsize=figsize)
    # обучающая кривая
    if train_sizes is not None:
        train_sizes_abs, train_scores, test_scores = learning_curve(model, X, y, train_sizes=train_sizes, scoring=scoring, cv=cv)
        result['learning_train_sizes'] = train_sizes_abs; result['learning_train_scores'] = train_scores; result['learning_test_scores'] = test_scores
        if plot == True:
            axs[0].plot(train_sizes_abs, train_scores.mean(axis=1), label='train_scores', color='blue', marker='o')
            axs[0].plot(train_sizes_abs, test_scores.mean(axis=1), label='test_scores', color='orange', marker='x')
            axs[0].set_title(f'learning curve - качество обучения от объема данных')
            axs[0].set_xlabel('train_sizes frac'); axs[0].set_ylabel(f'AVG {scoring}'); axs[0].legend(); axs[0].grid()
    # валидационная кривая
    if param_name is not None:
        train_scores, test_scores = validation_curve(model, X, y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)
        result['validation_train_scores'] = train_scores
        result['validation_test_scores'] = test_scores
        if plot == True:
            axs[1].plot(param_range, train_scores.mean(axis=1), label='train_scores', color='blue', marker='o')
            axs[1].plot(param_range, test_scores.mean(axis=1), label='test_scores', color='orange', marker='x')
            axs[1].set_title(f'validation curve - качество обучения от {param_name}')
            axs[1].set_xlabel(param_name); axs[1].set_ylabel(f'AVG {scoring}'); axs[1].legend(); axs[1].grid()
    return result

### КЛАСТЕРНЫЙ АНАЛИЗ
def find_clusters(X, method='kmeans', n_clusters=3, eps=0.5, N=5, scaler=True):
    """
    Кластеризация данных X (df.values) через dbscan или kmeans
    scaler - нужно ли перед этим масштабировать признаки для кластеризации (по умолчанию полезно)
    n_clusters - заданное кол-во кластеров для kmeans
    eps, N - окрестность и мин число семплов для dbscan
    return: лейблы кластеров и метрики качества алгоритмов
    """

    inertia, silhouette, ch_score = None, None, None
    if scaler == True:
        X = StandardScaler().fit_transform(X)

    if method == 'kmeans':
        label_cnt = n_clusters
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(X) # метки кластеров
        inertia = kmeans.inertia_
        if label_cnt > 1:
            silhouette = silhouette_score(X, labels)
            ch_score = calinski_harabasz_score(X, labels)


    elif method == 'dbscan':
        dbscan = DBSCAN(eps=eps, min_samples=N)
        labels = dbscan.fit_predict(X) # метки кластеров, -1 = шум
        label_cnt = np.unique(labels[labels!=-1]).shape[0]
        if label_cnt > 1:
            silhouette = silhouette_score(X[labels != -1], labels[labels != -1])
            ch_score = calinski_harabasz_score(X[labels != -1], labels[labels != -1])

    metrics = {'inertia' : inertia, 'silhouette' : silhouette, 'ch_score' : ch_score, 'label_cnt' : label_cnt}
    return labels, metrics

def apply_2d_map(X, scaler=True, model_type = 'umap', n_neighbors=15, min_dist=0.1, random_state=None):
    """
    Применяем метод понижения размерности до 2 с сохранением относительного расстояния
    для визуализации элементов множества
    model_type = umap, tsne; umap более продвинутый и быстрый графовый метод
    (n_neighbors, min_dist) - параметры для umap; n_neighbors выше - микро-тренды, ниже - макро
    X - датасет df.value; y - метка, для цветовой визуализации (если есть)
    scaler = флаги предварительной нормализации
    """
    if scaler == True:
        X = StandardScaler().fit_transform(X)
    if model_type == 'umap':
        model = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    else:
        model = TSNE(n_components=2, random_state=random_state)
    X_map = model.fit_transform(X)
    # plt.figure(figsize=(10, 8)); scatter = plt.scatter(X_map[:, 0], X_map[:, 1], c=y_sample, cmap='tab10', s=10, alpha=0.7)
    # plt.colorbar(scatter, ticks=range(10), label="Цифры")
    return X_map

def apply_pca(X, n_components = 'mle', n_components_range = None, scaler=True):
    """
    Применение PCA для понижение размерности датасета фичей X = df_x.values
    Для каждого кол-ва компонент максимизируется evr = explained_var_ratio
    scaler: нормализовать ли данные перед понижением размерностей (true по умолчанию)
    n_components_range (list) - если задан, то возвращает df_evr = n_components, evr
    n_components:
        'mle' - ищет через макс правдоподобие мин. размерность балансируя evr (оч консервативно)
        r=(0, 1) - найдет мин. кол-во компонент сохраняя evr = r
        int - найдет мин кол-во компонент равное этому числу
    return:
    - df_evr (если задан список перебора)
    - X_transform
    - evr_transform
    """
    if scaler:
        X = StandardScaler().fit_transform(X)

    # режим перебора значений
    df_evr = None
    if n_components_range is not None:
        explained_variance = []
        for n in n_components_range:
            pca = PCA(n_components=n)
            pca.fit(X)
            explained_variance.append(np.sum(pca.explained_variance_ratio_))
        df_evr = pd.DataFrame({'n_components' : n_components_range, 'evr' : explained_variance})

    # конкретное преобразование
    pca = PCA(n_components=n_components)
    X_transform = pca.fit_transform(X)
    evr = np.sum(pca.explained_variance_ratio_)
    return X_transform, evr, df_evr

def get_pca_and_features_relation(dfx, n_components = None, scaler=True):
    """Преобразуем фичи dfx = x1, x2, ... в пространство главных компонент pc1, pc2, ...
    с той же размерностью (сохраняя полную вариацию данных)
    Оцениваем важность каждой pc1 и физическую взаимосвязь с x1, x2, ...
    """
    if scaler:
        X = StandardScaler().fit_transform(dfx)
    if n_components == None:
        pca = PCA()
    else:
        pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    df_relation = pd.DataFrame(pca.components_.T,  columns=component_names, index=dfx.columns)
    df_relation.loc['evr'] = pca.explained_variance_ratio_
    dfx_pca = pd.DataFrame(X_pca, columns=component_names)
    return df_relation, dfx_pca

### ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ МОДЕЛИ
def get_shap_vals(model, X_test, plot=False):
    """
    SHAP - оценка вклада разных признаков в таргет модели (!!! логит для классификаторов, предикт для регрессоров)
    model - обученная модель, X_test - датасет для предсказания величин
    shap_vals - датасет для каждого семпла и фичи дающий shap (по сути вес) данного признака в предикт по семплу"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    if plot:
        # цвет = величина признака (краснее = больше), X = shap_val - чем больше, тем больше вклад в таргет>0, y - фичи
        shap.summary_plot(shap_values, X_test)
    return shap_values

def draw_sample_shap(shap_values, sample_num, max_display=10):
    """используем для отображения shap-диаграммы конкретного тестового семпла чтобы понять из чего складывается предсказание"""
    shap.waterfall_plot(shap_values[sample_num], max_display=max_display)

def draw_pdp(model, X_train, feature_names, ncols=None, nrows=1, figsize=(12, 4)):
    """Рисуем зависимость avg(model_predict) от конкретных feature при фиксированных остальных
    Для линейной регрессии эта зависимость линейная, для других моделей - более сложная (обобщение)
    model - обученная модель, X_train - датафрем pd с обучающими семплами, feature_names - какие фичи хотим строить"""

    feature_nums = [X_train.columns.get_loc(j) for j in feature_names]
    if ncols is None:
        ncols = len(feature_nums)
    _, ax = plt.subplots(ncols = ncols, nrows=nrows, figsize=figsize)
    pdp.from_estimator(model, X_train, features=feature_nums, feature_names=feature_names, ax=ax)
