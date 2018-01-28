import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.formula.api as smf
from causalinference import CausalModel
from scipy.stats import chisquare
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import tt_ind_solve_power
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
from joblib import Parallel, delayed

####### СТАТИСТИЧЕСКИЕ ТЕСТЫ
def fraq_to_list(num, denum):
    """преобразуем дробь num/denum в массив 0 0 0 1 ... для расчетов в стат тестах"""
    return np.append(np.repeat(1, num), np.repeat(0, denum - num))

def get_normal_list(mu, std, size):
    """Получение нормально распределенной метрики - для эмуляций; size = None -> return val else return list"""
    return np.random.normal(mu, std, size)

def ttest_calc(metric_1, metric_2 = None, alpha = 0.05, alternative = 'two-sided'):
    """
    https://habr.com/ru/articles/807051/
    Применение T-распределения для оценки дов интервалов и гипотез о равенстве средних
    Условия применимости: средневыборочное метрик распределено нормально (нужно проверить отдельно - см Валидация)
    p_value >= alpha -> approve H0 else reject.
    metric_1, 2 = списки; alpha = ошибка первого рода
    alternative = 'two-sided', 'greater', 'less' - проверка альтернативной гипотезы (и односторонний дов интервал)
    пример: alternative = greater; h0: Y - X <= 0; h1: Y - X > 0; p_val > alpha -> h0
    PS. нельзя комбинировать дов. интервалы greater/less -> растет ошибка множественного сравнения.
    PSS. при сравнении с реализациями вроде stats.ttest_ind важно учитывать что там проверяется не (Y-X), а (X-Y)
    return: conf_int_mean_diff, p_value, alternative, conf_int_mean_relative_diff
    """
     # кол-во степеней свободы в общем случае std1 != std2 (Welch test) -> df
    def welch_df(s1, s2, n1, n2):
        return ( (s1**2 / n1 + s2**2 / n2) ** 2 /
                ( ( (s1**2 / n1) ** 2 / (n1 - 1) ) +
                ( (s2**2 / n2) ** 2 / (n2 - 1) ) ) )

    # оценка доверительных интервалов для относительного эффекта (Y-X)/X по дельта-методу
    # подробнее см https://habr.com/ru/companies/avito/articles/571094/
    # грубая оценка confint_abs/avg(control) обычно более узкая, чем реальная
    # чем больше размер сравниваемых выборок (N>10**4) - тем ближе все оценки к реальному (bootstrap)
    # Расчет реализован только для критерия two-sided когда заданы две сравниваемые выборки
    def conf_rel_diff_delta_method(control, test, alpha):
        mean_control = np.mean(control)
        var_mean_control  = np.var(control) / len(control)
        difference_mean = np.mean(test) - mean_control
        difference_mean_var  = np.var(test) / len(test) + var_mean_control
        covariance = -var_mean_control
        relative_mu = difference_mean / mean_control
        relative_var = difference_mean_var / (mean_control ** 2) + var_mean_control * ((difference_mean ** 2) / (mean_control ** 4))\
                        - 2 * (difference_mean / (mean_control ** 3)) * covariance
        relative_distribution = stats.norm(loc=relative_mu, scale=np.sqrt(relative_var))
        left_bound, right_bound = relative_distribution.ppf([alpha/2, 1-alpha/2])
        pvalue = 2 * min(relative_distribution.cdf(0), relative_distribution.sf(0))
        return (left_bound, right_bound), pvalue

    conf_int_mean_relative_diff = None # считается только для two-sided + две метрики
    if metric_2 is None: # оценка среднего одной метрики
        mean_diff = np.mean(metric_1)
        se = np.sqrt(np.std(metric_1) ** 2 / len(metric_1))
        df = len(metric_1) - 1
    else: # оценка разницы средних двух метрик
        mean_diff = np.mean(metric_2) - np.mean(metric_1)
        se = np.sqrt(np.std(metric_1) ** 2 / len(metric_1) + np.std(metric_2) ** 2 / len(metric_2))
        df = welch_df(np.std(metric_1), np.std(metric_2), len(metric_1), len(metric_2)) # общий случай std_1 != std_2

    if alternative == 'two-sided': # h1: mean_diff != 0
        t = stats.t.ppf(1 - alpha / 2, df=df)
        cdf_stat = stats.t.cdf(np.abs(mean_diff / se), df=df)
        p_val = 2 * (1 - cdf_stat) # p_val > alpha -> h0: mean_diff = 0
        confint = mean_diff + np.array([-1, 1]) * t * se
        if metric_2 is not None:
            conf_int_mean_relative_diff = conf_rel_diff_delta_method(metric_1, metric_2, alpha)

    elif alternative == 'less': # h1: mean_diff < 0
        t = stats.t.ppf(1 - alpha, df=df)
        cdf_stat = stats.t.cdf(mean_diff / se, df=df)
        p_val = cdf_stat # p_val > alpha -> h0: mean_diff >= 0
        confint = (-np.inf, mean_diff + t * se) # confint[1] < 0 -> h1

    elif alternative == 'greater': # h1: mean_diff > 0
        t = stats.t.ppf(1 - alpha, df=df)
        cdf_stat = stats.t.cdf(mean_diff / se, df=df)
        p_val = 1 - cdf_stat # p_val > alpha -> h0: mean_diff <= 0
        confint = (mean_diff - t * se, np.inf) # confint[0] > 0 -> h1

    return confint, p_val, alternative, conf_int_mean_relative_diff

def bootstrap_calc(metric_1, metric_2 = None, stat_func = np.mean, iter = 10**4, alpha = 0.05, diff_type = 'abs', adj_small = False):
    """Использование семплирования для оценки разницы статистик stat_func(X) на двух или одной метрике
    test_func -> это может быть среднее, медиана или др статистики
    diff_type = 'abs', 'rel'; для abs: diff = Y-X; для rel: diff = (Y-X)/X; работает только для двух выборок
    adj_small = корректировка смещения бутстрапа на малых выборках, есть смысл включать при len(metric) < 100-200
    return confint_test_stat_diff, p_value
    """
    boot_list = [] # sampling
    if metric_2 is None:
        for _ in range(iter):
            stat_ = stat_func(np.random.choice(metric_1, len(metric_1), replace = True))
            boot_list.append(stat_)
    else:
        for _ in range(iter):
            s1 = stat_func(np.random.choice(metric_1, len(metric_1), replace = True))
            s2 = stat_func(np.random.choice(metric_2, len(metric_2), replace = True))
            if diff_type == 'abs':
                diff = s2 - s1
            elif diff_type == 'rel':
                diff = (s2 - s1) / s1
            boot_list.append(diff)

    if adj_small:
        # оценка по исходной выборке
        s1, s2 = stat_func(metric_1), stat_func(metric_2)
        if diff_type == 'abs':
                diff = s2 - s1
        elif diff_type == 'rel':
                diff = (s2 - s1) / s1
        # часто разница средних для семплируемой статистики stat_func хорошая оценка смещения
        # но иногда лучше использовать для оценки сдвига другие функции например разница между квантилями для исходной выборки и семплирования
        bias = np.mean(boot_list) - diff
        boot_list = [j - bias for j in boot_list]

    confint = np.percentile(boot_list, 100 * (alpha / 2)), np.percentile(boot_list, 100 * (1 - alpha / 2))
    q_ = stats.norm.cdf(x=0, loc=np.mean(boot_list), scale=np.std(boot_list, ddof=1)) # p val calc
    p_val = q_ * 2 if 0 < np.mean(boot_list) else (1 - q_) * 2
    return confint, p_val

def bootstrap_poisson_calc(metric_1, metric_2, n_iter = 10**4, X1 = None, X2 = None, alpha = 0.05):
    """Оцениваем разницу средних взвешивая на базе распределения Пуассона
    Работает корректно для больших выборок len(metric)>10**3
    Веса X1,2 ~ Poisson(1) могут быть сгенерированы и записаны в память заранее
    Если их нет, то генерируются внутри функции
    """
    # считаем заранее либо в моменте
    if X1 is None:
        X1 = np.random.poisson(lam=1, size=(n_iter, len(metric_1)))
        X2 = np.random.poisson(lam=1, size=(n_iter, len(metric_2)))
    # матричный расчет списка средних
    mu1 = np.dot(X1, metric_1) / X1.sum(axis=1)
    mu2 = np.dot(X2, metric_2) / X2.sum(axis=1)
    boot_list = mu2 - mu1
    # возвращаем дов интервалы для разницы средних
    return np.percentile(boot_list, [100 * alpha/2, 100 - alpha/2])

def ttest_stratification_calc(df_con, df_test, weights, alpha = 0.05):
    """"Стратифицированный T-test для двух выборок
    df_con/test = Y, S = целевая метрика, значение страты
    например Y = выручка, S = название региона;
    weights = удельный вес каждой страты на исторических данных df_prev
    weights = df_prev.S.value_counts(normalize=True)
    return: confint, p_val
    """
    con_size = len(df_con.Y)
    test_size = len(df_test.Y)
    con_avg = (df_con.groupby('S')['Y'].mean() * weights).sum()
    test_avg = (df_test.groupby('S')['Y'].mean() * weights).sum()
    con_var = ((df_con.groupby('S')['Y'].var()) * weights).sum()
    test_var = ((df_test.groupby('S')['Y'].var()) * weights).sum()
    # test calc
    delta_avg = test_avg - con_avg
    se = np.sqrt((con_var / con_size + test_var / test_size))
    z = stats.t.ppf(1 - alpha / 2, df=(con_size + test_size - 2))
    p_val = 2 * (1 - stats.t.cdf(np.abs(delta_avg / se), df=(con_size + test_size - 2)))
    confint = delta_avg + np.array([-1, 1]) * z * se
    return confint, p_val

def multitest_calc(p_value_list, alpha = 0.05, alpha_type = 'fwer'):
    """Корректировка p_values по группе парных ab тестов с различным контролем ошибки
    https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
    p_value_list = список p_value всех проверяемых гипотез (например, список результатов всех T-тестов для N сравнений)
    alpha = уровень контроля значимости fdr, fwer в зависимости от alpha_type
    alpha_type = fdr, fwer (выбраны оптимальные критерии для каждого из типов контроля)
    FWER = family wise error rate = контроль ошибки "хотя бы один ложный прокрас из всех тестов". Строгое условие
    FDR = false discovery rate = контроль доли ложных прокрасов среди прокрасов - мягкое условие (для поиска положительных инсайтов)
    return: список решений по тестам на уровне alpha; p_val_corrected; alpha_adj - уменьшенное на множественную поправку
    """
    # holm = аналог Бонферрони, но мощнее; fdr_bh = Benjamini/Hochberg оптимален для контроля FDR
    decision_list, p_val_corrected, alpha_adj, _ = multipletests(p_value_list, alpha = alpha, method=alpha_type.replace('fwer','holm').replace('fdr', 'fdr_bh'))
    return decision_list, p_val_corrected, alpha_adj


####### ТОЧНОСТЬ И КОРРЕКТНОСТЬ КРИТЕРИЕВ
def get_mde_detail(control, n_branch = 2, ratio = 1, alpha = 0.05, power = 0.8):
    """
    ratio = len(control) / len(exp) - изменяется когда например катим ассиметричные тесты 70% теста итд
    n_branch - если запускается несколько веток - учитывается поправка Бонферрони (n_branch кол во веток УЧИТЫВАЯ контроль)
    return mde info; mde - минимально детектируемый эффект с заданными ошибками первого и второго рода
    """
    comparison = n_branch * (n_branch - 1) / 2
    N_c = len(control)
    control_mean, control_std = np.mean(control), np.std(control)
    delta = control_std * tt_ind_solve_power(nobs1 = N_c, alpha= alpha / comparison, power = power, ratio = ratio, alternative='two-sided')
    return f"""Участников в контроле {len(control)}, среднее: {round(control_mean, 2)}, mde_abs = {round(delta, 3)}; mde_rel = {round(100 * delta / control_mean, 1)}%"""

def get_mean_diff_confint_width(control, alpha = 0.05):
    """Ширина доверительного интервала для разности средних со схожими дисперсиями"""
    ci_width = stats.t.ppf(1 - alpha / 2, df=(len(control) - 1)) * np.std(control, ddof=1) / np.sqrt(len(control)) # для среднего по выборке
    # среднее для разности больше в sqrt(2) т к SE_diff = sqrt(SE_control**2 + SE_control**2)
    return np.sqrt(2) * ci_width

def check_branch_balance(f_real, f_exp = None):
    """Проверка соответствия реальных частот f_real (напр, кол-во участников в ветках экспа [105, 115])
    ожидаемым математически частотам [100, 100]
    Если f_exp = None, считаем ожидаемое распр. равномерным f_exp = sum(f_obs)/len(f_obs))
    """
    _, p_value = chisquare(f_real, f_exp)
    text = 'branch sizes valid'
    if p_value < 0.05:
        text = 'branch sizes invalid'
    return text + f' p_val = {round(p_value, 4)}'

def validation_ttest(X, backets_cnt = 10, aa_test_cnt = 10**4, alpha = 0.05):
    """
    Проводим валидацию метрики X, через семплирование АА-тестами. Алгоритм:
    1) метрика X разбивается на backets_cnt одинаковых частей (четное!)
    2) получаем попарные backets_cnt/2 АА-тестов -> вычисляем p_val
    3) перемешиваем метрику и так повторяем далее, пока не наберем iter_cnt тестов
    оцениваем FPR (долю ложных прокрасов); дополнительно семплируем средневыборочное X, проверяем его на нормальность
    PS. backets_cnt рекомендуется выбирать из условия len(X)/backets_cnt ~ real_control
    """
    if backets_cnt % 2 > 0:
        return 'выберите четное число backets_cnt'
    iter = int(2 * aa_test_cnt / backets_cnt)
    p_val_list = []
    for _ in range(iter):
        np.random.shuffle(X) # inplace
        aa_test_list = np.array_split(X, backets_cnt) # попарные АА тесты
        for j in [i for i in range(0, backets_cnt - 1) if i % 2 == 0]:
            p_val = stats.ttest_ind(aa_test_list[j], aa_test_list[j+1])[1]
            p_val_list.append(p_val)
    # оцениваем FPR
    fpr_list = [1 if j <= alpha else 0 for j in p_val_list]
    fpr, _ = ttest_calc(fraq_to_list(sum(fpr_list), len(fpr_list))) # дов интервалы для ошибки 1-го рода
    # семплируем средневыборочное; оцениваем нормальность
    avg_list = []
    for _ in range(iter):
        avg_ = np.mean(np.random.choice(X, int(len(X) / backets_cnt), replace=False))
        avg_list.append(avg_)
    _, p_value = stats.shapiro(avg_list)
    decision = ['INVALID' if p_value <= alpha else 'VALID'][0]
    ans = f"""
    Кол-во АА-тестов {len(p_val_list)}; Backets_cnt = {backets_cnt};
    AA_control_size ~ {int(len(X)/backets_cnt)}; AA_control_avg ~ {round(np.mean(X), 5)}
    AA_FPR = {np.round(100 * fpr, 3)}%
    Нормальность средневыборочн. {decision}, p_val_shapiro ~ {round(p_value, 6)}"""
    # при необходимости можно визуализировать списки через get_visualisation
    return ans, p_val_list, avg_list

def simulate_test(stat_test,
                  effect_type = 'loc',
                  effect_range = [0, 1, 2, 3],
                  metric_dist = 'normal',
                  metric_size = 500,
                  metric_cnt = 2,
                  metric_dist_params = {'scale' : 1},
                  iter_cnt = 500,
                  parallel = True):
    """Симуляция критерия stat_test с поиском ошибок первого и второго рода на синтетических или реальных данных
    effect_type - название параметра который смещаем для эмуляции a/b (например матожидание; loc)
    effect_range - список смещений (когда 0 - режим a/a теста и проверяем ошибку первого рода)
    metric_dist = normal, exponential, uniform или кастомное (см пример в доке ниже metric_dist_example)
    metric_size = размер сравниваемых веток
    metric_cnt = кол-во сравниваемых веток (2 = a/b)
    metric_dist_params = фикс параметры распределений (их не варьируем)
    iter_cnt = кол-во симуляций на каждое смещение
    parallel = распараллеливаем вычисление по ядрам
    return df = ср и дов интервалы для ошибок 1 и 2 рода для каждого смещения
    """

    # в противном случае передается кастомный metric_dist()
    if metric_dist == 'normal':
        metric_dist = np.random.normal
    if metric_dist == 'exponential':
        metric_dist = np.random.exponential
    if metric_dist == 'uniform':
        metric_dist = np.random.uniform

    pos_res = []; pos_res_conf = []; neg_res = []; neg_res_conf = []
    for eff in effect_range:

        # изготовим большой батч данных, затем будем нарезать его на тесты
        # a/a (n-1 ветка)
        metric_dist_params[effect_type] = 0
        metric_dist_params['size'] = (iter_cnt * metric_size, metric_cnt - 1)
        metric_arr = metric_dist(**metric_dist_params)
        # a/b (последняя ветка выборки)
        metric_dist_params['size'] = iter_cnt * metric_size
        metric_dist_params[effect_type] = eff
        eff_metric = metric_dist(**metric_dist_params)
        # стекаем все вместе
        metric_arr = np.column_stack((metric_arr, eff_metric))
        # должен быть metric_arr.shape = (iter_cnt * metric_size, metric_cnt)
        # при кастомных метриках metric_dist размер вдоль X может отличаться - семплируем корректно:
        if metric_arr.shape[0] != iter_cnt * metric_size:
            idx = np.random.choice(metric_arr.shape[0], size=iter_cnt * metric_size, replace=False)
            metric_arr = metric_arr[idx]
        # массив тестов
        chunk_list = np.vsplit(metric_arr, iter_cnt)
        res = []
        if parallel:
            res = Parallel(n_jobs=-1)(delayed(stat_test)(np.hsplit(chunk, metric_cnt)) for chunk in chunk_list)
        else:
            for chunk in chunk_list:
                res.append(stat_test(np.hsplit(chunk, metric_cnt))) # input = массив метрик [X, Y, ...]

        # считаем средние и дов интервалы
        res_inv = [1 if j == 0 else 0 for j in res]
        pos_res.append(100 * np.mean(res)); pos_res_conf.append(np.round(100 * ttest_calc(res)[0], 2))
        neg_res.append(100 * np.mean(res_inv)); neg_res_conf.append(np.round(100 * ttest_calc(res_inv)[0], 2))

    df_stat = pd.DataFrame({'effect_shift' : effect_range,
                            'type I error' : np.round(pos_res, 2), 'type I error conf' : pos_res_conf,
                             'type II error' : np.round(neg_res, 2), 'type II error conf' : neg_res_conf })
    df_stat.loc[df_stat.effect_shift == 0, 'type II error'] = ''; df_stat.loc[df_stat.effect_shift == 0, 'type II error conf'] = ''
    df_stat.loc[df_stat.effect_shift > 0, 'type I error'] = ''; df_stat.loc[df_stat.effect_shift > 0, 'type I error conf'] = ''
    return df_stat

def stat_test_example(metric_arr):
    alpha = 0.05
    p_value = ttest_calc(metric_arr[0], metric_arr[1], alpha=alpha)[1]
    return int(p_value < alpha)

def metric_dist_example(scale, loc, size):
    # примесь выбросов = 10%
    if isinstance(size, tuple): # a/a batch
        size_outl = (int(0.1 * size[0]), size[1])
    else: # a/b branch
        size_outl = int(0.1 * size)
    return np.append(np.random.normal(loc = loc, scale=scale, size=size), np.random.normal(loc=500, size=size_outl))


####### ПОВЫШЕНИЕ ЧУВСТВИТЕЛЬНОСТИ
def cuped_calc(df, x = ['Y_prev'], y = 'Y', T = 'exp_group', method = 'ols', df_prev = None):
    """
    CUPED =  Controlled-experiment Using Pre-Experiment Data (при использовании ML -> расширение до CUPAC)
    df - целевой экспериментальный датасет; y - целевая переменная; T - индикатор тестовой группы control/experiment или 0/1
    x - список уточняющих ковариат = ['x1', 'x2' ...]
    method - модель обучения = ols (лин регрессия), hgb (HistGradientBoostingRegressor), etr (ExtraTreesRegressor)
    df_prev - исторические данные; если заданы - то обучение идет на них, иначе - на контроле
    return: Y_adj - скорректированная целевая метрика; avg(Y) = avg(Y_adj); std(Y_adj) < std(Y)
    """
    df = df.copy()
    df[T] = df[T].replace({'control' : 0, 'experiment' : 1}) # дополнительная подготовка
    # 1. model fit
    if df_prev is None:
        tmp_fit = df[df[T] == 0]         # обучаем модель на контрольных данных
    else:
        tmp_fit = df_prev # берем исторические данные для обучения
    if method == 'ols':
        model = smf.ols(f'{y} ~ ' + ' + '.join(x), data=tmp_fit).fit()
    elif method == 'hgb':
        Y_ = tmp_fit[y].values
        X_ = tmp_fit[x].values
        model = HistGradientBoostingRegressor().fit(X_, Y_)
    elif method == 'etr':
        print('высокий риск переобучения на контроле! использовать на df_prev')
        Y_ = tmp_fit[y].values
        X_ = tmp_fit[x].values
        model = ExtraTreesRegressor().fit(X_, Y_)
    # 2. моделируем Y_forecast по ковариатам для коррекции целевой метрики
    X_ = df[x]; Y_forecast  = model.predict(X_)
    # контроль переобучения - полученный прогноз должен быть независим от T во избежание смещения оценки
    print(f'correlation Treatment - Y_forecast {np.corrcoef(df[T].values, Y_forecast)[0][1]}')
    # 3. create Y_adj: Y ~ Y_forecast -> Y_adj
    Y = df[y].values; Y_adj = Y - (Y_forecast - np.mean(Y_forecast))
    return Y_adj

def cuped_simple(Y, Y_cov):
    """
    Получаем Y_adj из предиктивной ковариаты (Y_cov = Y_prev или берем из ML модели)
    Главное условие np.cov(exp_group, Y_cov) = 0, независимость сплитования и ковариты
    """
    theta = np.cov(Y_cov, Y)[0, 1] / np.var(Y_cov)
    Y_adj = Y - theta * (Y_cov - np.mean(Y_cov))
    return Y_adj

####### ПРЕОБРАЗОВАНИЯ МЕТРИК
# бакетный анализ
def ratio_linearisation(metric_num, metric_denum):
    """
    Линеаризация Ratio-метрики типа R = metric_num / metric_denum -> linear_user_metric
    Возможно разными способами:
    - дельта-метод (более простой в вычислении, менее точный; нет поюзерной метрики)
    - линеаризация (разложение метрики до первого члена, быстрее но менее точно)
    - тейлор-2 (разложение метрики по производным до второго члена, точнее но дольше)
    здесь мы применяем линеаризацию с помощью разложения тейлора до первого порядка
    X/Y ~ mx/my + 1/my * (X - mx) - mx/my**2 * (Y - my) = mx/my + 1/my(X - Y*mx/my)
    https://habr.com/ru/company/avito/blog/454164/
    """
    mx, my = np.mean(metric_num), np.mean(metric_denum)
    return mx / my + (1 / my) * (metric_num - metric_denum * (mx / my))

def outlier_fix(metric, thr = 99, thr_type = 'ptl', fix_type = 'd'):
    """
    Обработка метрики от имеющихся в ней выбросов
    thr_type = ptl (percentile), val (real_val) = отсечение по процентам или реальному порогу
    fix_type = d (drop), w (winsor) = убираем выбросы, заменяем на максимально допустимые (винсоризация)
    """
    metric = np.array(metric)
    if thr_type == 'ptl':
        thr = np.percentile(metric, thr)
    thr_real = max(metric[metric <= thr])
    if fix_type == 'd':
        res = metric[metric <= thr_real]
    elif fix_type == 'w':
        res = np.array([j if j <= thr_real else thr_real for j in metric])
    return res

####### ЛИНЕЙНАЯ РЕГРЕССИЯ
def ols_calc(df, y = 'Y', x = ['X'], smf_formula = None, weight = None, regul_alpha = None, regul_type = 'L1'):
    """
    OLS = ordinary least square
    df = датасет с целевой метрикой Y и ковариатами x = ['x1', 'x2', ...]
    smf_formula = smf-формула в R-нотации. Если указана, то значения y/x игнорируются; работает когда regul_alpha NULL
    regul_alpha; regul_type = L1/L2 = параметры для регуляризации
    L1: штраф за высокие |k1|+|k2|+...; L2: за k1**2 + k2**2 + ...
    PS. регуляризация усложняет оценку дов интевалов -> их можно считать только с bootstrap
    weight = колонка весов для каждого наблюдения - применяется только в OLS без регуляризации
    return: info = датасет с информацией по регрессии, model = обученная модель для predict
    """
    if regul_alpha is not None:
        # регуляризация; регрессия методами sklearn
        if regul_type == 'L1':
            model = Lasso(alpha = regul_alpha)
            model.fit(df[x], df[y])
            info = pd.DataFrame({'feature' : model.feature_names_in_,
                                 'coef' : model.coef_})
            info['R^2'] = r2_score(df[y], model.predict(df[x]))
            info['regul_type'] = regul_type
            info['regul_alpha'] = regul_alpha

        elif regul_type == 'L2':
            model = Ridge(alpha = regul_alpha)
            model.fit(df[x], df[y])
            info = pd.DataFrame({'feature' : model.feature_names_in_,
                                 'coef' : model.coef_})
            info['R^2'] = r2_score(df[y], model.predict(df[x]))
            info['regul_type'] = regul_type
            info['regul_alpha'] = regul_alpha
    else:
        # регрессия через statsmodel (больше доп инфы)
        if smf_formula is None:
            smf_formula = f'{y} ~ ' + ' + '.join(x)
        if weight is None:
            model = smf.ols(smf_formula, data = df).fit()
        else:
            weights = df['weight'].values
            model = smf.wls(smf_formula, data = df, weights = weights).fit()
        info = pd.DataFrame(model.summary().tables[1])
        new_columns = info.iloc[0]
        info.columns = [str(j) for j in new_columns]
        info = info[1:].reset_index(drop=True).rename(columns={'' : 'feature',
                                                             '[0.025' : 'conf_int1',
                                                             '0.975]' : 'conf_int2',
                                                             'P>|t|' : 'pval',
                                                             'std err' : 'std_'})
        info['R^2'] = model.rsquared
    # финальный результат - dataframe + fit model
    return info, model

###### ПРИЛОЖЕНИЕ: дополнительные способы оценки относительных дов интервалов для T-распределения
def conf_rel_diff_cohavi(X, Y, alpha = 0.05):
    # оценка относительного эффекта по Kohavi - приближенное разложение
    # https://www.researchgate.net/publication/220451900_Controlled_experiments_on_the_web_Survey_and_practical_guide
    # дает схожий эффект что и при дельта методе (который реализован в ttest_calc)
    def welch_df(s1, s2, n1, n2):
        return ( (s1**2 / n1 + s2**2 / n2) ** 2 /
                ( ( (s1**2 / n1) ** 2 / (n1 - 1) ) +
                ( (s2**2 / n2) ** 2 / (n2 - 1) ) ) )
    df = welch_df(np.std(X), np.std(Y), len(X), len(Y))
    t = stats.t.ppf(1 - alpha / 2, df=df)
    mean1, mean2 = np.mean(X), np.mean(Y)
    std1 = np.std(X) / np.sqrt(len(X))
    std2 = np.std(Y) / np.sqrt(len(Y))
    cv1, cv2 = std1 / mean1, std2 / mean2
    rel_diff = (mean2 - mean1) / mean1
    conf_rel_diff = (rel_diff + 1) * (1 + np.array([-1, 1]) * t * np.sqrt(cv1**2 + cv2**2 - t**2 * cv1**2 * cv2**2) / (1 - t * cv1 ** 2)) - 1
    return conf_rel_diff
