import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.formula.api as smf
from causalinference import CausalModel
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns

####### CAUSAL INFERENCE
def calc_psm_eff(df, X, T='treatment', Y='y'):
    """
    Propensity Score Matching
    df = массив с данными; Y = целевая метрика; X = ковариаты; T = сплит переменная в формате 0;1
    causalmodel используем для поиска через KNeighborsRegressor ближайшего соседа по ps
    также идет bias_adj неидеального матчинга с помощью линейной регрессии
    return ATE - эффект от эксперимента на целевую метрику Y
    """
    # ps = propensity score = вероятность принадлежать классу test (1)
    df['ps'] = LogisticRegression(C=1e6).fit(df[X], df[T]).predict_proba(df[X])[:, 1]
    # матчинг, используя kNN на переменной ps - для каждого контрольного семпла ищем самый ближний по ps из теста
    cm = CausalModel(
            Y=df[Y].values,
            D=df[T].values,
            X=df.ps.values
        )
    cm.est_via_matching(matches=1, bias_adj=True)
    # выводит average treatment effect после матчинга
    return cm.estimates['matching']['ate']

def calc_iptw_eff(df, X, T='treatment', Y='y'):
    """
    Аналогичная PSM оценка на сбалансированной через IPTW метод
    Здесь мы взвешиваем семплы (weights) в зависимости от их "экзотичности" - тем самым выправляя баланс выборок
    return: ATE - эффект от эксперимента на целевую метрику Y; дов интервалы можно оценивать через bootstrap
    [iptw(df.sample(frac=1, replace=True), X) for j in range(iter)]
    """
    df['ps'] = LogisticRegression(C=1e6).fit(df[X], df[T]).predict_proba(df[X])[:, 1]
    # IPTW calc
    weight = ((df[T] - df.ps) / (df.ps * (1 - df.ps)))
    return np.mean(weight * df[Y])

def cohen_d(d1, d2):
    """
    Нормированное на дисперсию расстояние между средними ковариаты
    для двух выборок 1 и 2. Используем для проверки сбалансированности
    выборок по ковариатам после матчинга.
    cohen<10% можно считать нормально сбалансированной
    """
    # d1,2 - проверяемая ковариата для 1 и 2 групп
    n1, n2 = len(d1), len(d2)
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    u1, u2 = np.mean(d1), np.mean(d2)
    return round(100 * (u2 - u1) / s, 2)

def check_cohen_stat(df1, df2, features = None, features_excl = None):
    """Проверяем насколько сбалансированы две сравниваемые выборки по признакам features
    features_excl = arr, список фичей которые исключаем из расчета
    return: df_cohen_stat, df_cohen_stat_inf (фичи с ошибками в расчете)
    """
    if features is None:
        features = df1.columns
    if features_excl is not None:
        features = [f for f in features if f not in features_excl]

    df_cohen_stat = pd.DataFrame(None, columns=['feature', 'avg1', 'avg2', 'cohen_d_percent']); j=0
    for f in features:
        f1, f2 = df1[f].values, df2[f].values
        try:
            f1_avg, f2_avg = np.round(np.mean(f1), 4), np.round(np.mean(f2), 4)
            c = cohen_d(f1, f2)
        except:
            c, f1_avg, f2_avg = np.inf, np.inf, np.inf
        df_cohen_stat.loc[j, :] = f, f1_avg, f2_avg, c; j+=1

    not_calc_features = df_cohen_stat[df_cohen_stat.cohen_d_percent.apply(abs) == np.inf].feature.values
    df_cohen_stat = df_cohen_stat[df_cohen_stat.cohen_d_percent.apply(abs) != np.inf]

    cohen_total = np.sqrt(sum([j**2 for j in df_cohen_stat.cohen_d_percent.values]))
    df_cohen_stat.loc[j, :] = 'total', '', '', cohen_total

    return df_cohen_stat, not_calc_features

def get_psm_df(df1, df2, features=None, features_excl=None, plot_overlap=False, ps_model=LogisticRegression()):
    """
    Для каждого объекта из df1 возвращаем объект из df2,
    наиболее близкий к df1 по propensity score (ps)
    ps_model - модель которая на объединенном датасете считает PS
    features - фичи, по которым происходит матчинг (по умолчанию все)
    plot_overlap - если True, то смотрим распределение ps в обеих группах (должно перекрываться)
    """
    if features is None:
        features = df1.columns
    if features_excl is not None:
        features = [f for f in features if f not in features_excl]

    df1, df2 = df1.copy(), df2.copy()
    df1['group'] = 0; df2['group'] = 1
    df = pd.concat([df1, df2], ignore_index=True)

    # ps = вероятность каждого семпла попасть в group = 1
    df['ps'] = ps_model.fit(df[features].values, df['group'].values).predict_proba(df[features].values)[:, 1]

    # смотрим перекрытие ps в обеих группах
    if plot_overlap == True:
        plt.grid()
        plt.title('check overlap')
        sns.histplot(data=df, x='ps', hue='group', stat="probability", bins=100)

    # для датасета group = 0 находим из датасета group=1 строчки максимально близкие по ps
    knn = KNeighborsRegressor(n_neighbors=1).fit(df[df.group == 1][['ps']], df[df.group == 1].index)
    nearest_neighbors_idx = knn.predict(df[df.group == 0][['ps']]).astype(int)
    return df.loc[nearest_neighbors_idx, :].drop(columns=['group', 'ps'])

def get_psm_df_sym(df1, df2, features=None, features_excl=None, plot_overlap=False, ps_model=LogisticRegression()):
    """Для PSM матчинг симметричный, т.е сначала подбираем df2_matched для df1, а затем наборот и уже по объединению оцениваем эффект
    Построено на базе функции get_psm_df
    """
    df2m = get_psm_df(df1, df2, features, features_excl, plot_overlap, ps_model) # подобрали df2m максимально близкий по ковариатам к df1
    df1m = get_psm_df(df2, df1, features, features_excl, plot_overlap, ps_model) # сделали подбор наоборот
    df1_combined = pd.concat([df1, df1m], ignore_index=True)
    df2_combined = pd.concat([df2m, df2], ignore_index=True)
    return df1_combined, df2_combined

def calc_did(con_before, con_after, exp_before, exp_after):
    """
    DIFF-IN-DIFF
    Расчет изменения метрики в ветке exp до/после из-за целевого влияния,
    DID = (exp_avg_after - exp_avg_before) - (control_avg_after - control_avg_before)
    Ключевое условие: параллельность трендов exp/control до целевого влияния (проверять отдельно)
    exp, con = arr; массив значений для семплов before/after целевого влияния (метрика)
    return: {DID, DID_p_val>0, DID_confint}
    """
    # собираем датасет из имеющихся данных
    df1 = pd.DataFrame({'metric' : exp_before}); df1['exp_group'] = 1; df1['dt'] = 0
    df2 = pd.DataFrame({'metric' : exp_after}); df2['exp_group'] = 1; df2['dt'] = 1
    df3 = pd.DataFrame({'metric' : con_before}); df3['exp_group'] = 0; df3['dt'] = 0
    df4 = pd.DataFrame({'metric' : con_after}); df4['exp_group'] = 0; df4['dt'] = 1
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    # обучаем регрессию
    model = smf.ols('metric ~ exp_group + dt + exp_group * dt', data=df).fit()
    ans = {}
    ans['did'] = model.params.loc['exp_group:dt']
    ans['did_confint'] = (model.conf_int().loc['exp_group:dt'][0], model.conf_int().loc['exp_group:dt'][1])
    ans['pval_did>0'] = model.pvalues.loc['exp_group:dt']
    return ans

def calc_rdd(metric, x, x_target, use_weights=True, c=0, h=1):
    """
    Regression Discounty Design
    Оценка эффекта на metric целевого влияния произошедшего в x_target
    metric - исследуемая метрика; x - независимая ковариата (например, время)
    считаем что при x<x0 целевого влияния не было, а при x>=x0 было
    Оцениваем скачок metric из-за влияния:
    rdd = metric(x=x0+e) - metric(x=x0-e) = delta_metric
    use_weights = использовать функцию,  штрафующую семплы при удалении от границы разрыва
    c, h = параметры функции: высота, ширина ядра
    return: {rdd, rdd_confint, p_val_rdd>0}
    """
    # смещаем "момент" целевого влияния в точку 0
    df = pd.DataFrame({'metric' : metric, 'x' : x})
    df['x'] = df['x'] - x_target
    df['D'] = (df.x > 0).astype(int)

    # обучаем модель с ядром, штрафующим влияние точек при удалении от границы разрыва
    def kernel(R, c, h):
        indicator = (np.abs(R-c) <= h).astype(float)
        return indicator * (1 - np.abs(R-c)/h)
    if use_weights:
        model = smf.wls("metric ~ x * D + D + x", df, weights=kernel(df.x.values, c=c, h=h)).fit()
    else:
        model = smf.wls("metric ~ x * D + D + x", df).fit()
    ans = {}
    ans['rdd'] = model.params.loc['D']
    ans['rdd_confint'] = (model.conf_int().loc['D'][0], model.conf_int().loc['D'][1])
    ans['pval_rdd>0'] = model.pvalues.loc['D']
    return ans
