import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

#### EDA = Exploratory Data Analysis - тулзы для лучшей визуализации и ресреча
###### Визуализация
def init_global_settings():
    matplotlib.rcParams['figure.titlesize'] = 15
    matplotlib.rcParams['font.size'] = 15
    matplotlib.rcParams['legend.fontsize'] = 10
    matplotlib.rcParams['axes.titlesize'] = 15
    matplotlib.rcParams['axes.labelsize'] = 15
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    matplotlib.rcParams['lines.linewidth'] = 2

def get_subplots(size=(15, 5), rows=1, cols=1,
                  ylabels = None, xlabels = None, titles = None, show_grid = True, constrained_layout = True):
    """Получаем на выходе матрицу графиков с преднастройкой форматов
    size = общая размерность матрицы
    rows, cols = кол-во колонок и строк матрицы графиков
    xlabels, ylabels, titles - сетки подписей для графика
    show_grid = рисовать ли сетку на полотнах
    constrained_layout = рисовать сетку графиков без перекрытий
    """
    def transform_to_dim2(arr):
        arr = np.array(arr)
        if arr.ndim == 1:
            arr = np.array([arr])
        return arr
    _, ax = plt.subplots(nrows=rows, ncols=cols, figsize=size, constrained_layout=constrained_layout)
    if rows == 1 and cols == 1:
        ax = np.array([[ax]])
    else:
        ax = transform_to_dim2(ax)
    for i in range(rows):
        for j in range(cols):
            if show_grid:
                ax[i, j].grid(True)
            if titles is not None:
                titles = transform_to_dim2(titles)
                ax[i, j].set_title(titles[i, j])
            if xlabels is not None:
                xlabels = transform_to_dim2(xlabels)
                ax[i, j].set_xlabel(xlabels[i, j])
            if ylabels is not None:
                ylabels = transform_to_dim2(ylabels)
                ax[i, j].set_ylabel(ylabels[i, j])
    if ax.shape == (1, 1):
        return ax[0, 0]
    elif ax.shape[0] == 1:
        return ax[0]
    else:
        return ax

def plot(x = None, y = None, ax = None, style='line', hue=None, size=None, palette = 'viridis'):
    """
    отображение графика на полотне ax;
    x, y зависимые величины (если x is None, то просто по порядку)
    style = 'line', 'scatter', ...
    hue, size = числовые отображения для цвета и размера точек графика
    """
    if x is None:
        x = np.arange(len(y))
    if style == 'line':
        sns.lineplot(x=x, y=y, ax=ax, hue=hue, size=size, palette=palette)
    elif style == 'scatter':
        sns.scatterplot(x=x, y=y, ax=ax, hue=hue, size=size, palette=palette)





def get_percentile_curve(val_list, per_start = 0, per_end = 100, per_step = 5, xlabel = '', ylabel = '', title = '', figsize=(10, 6)):
    """
    Перцентильная кривая = аналог гистограммы
    Берем метрику val_list и строим график вида: какой процент семплов (ось X) меньше значения Y
    """
    per_range = list(range(per_start, per_end + per_step, per_step))
    per_list = np.percentile(val_list, per_range)
    plt.figure(figsize=figsize)
    plt.plot(per_range, per_list)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(); plt.xticks(per_range); plt.show()

def histogram_visualise(X, bins = 50, xlabel = '', ylabel = '', title = '', figsize=(10, 6)):
    """Удобная визуализация для гистограммы с надписями"""
    plt.figure(figsize=figsize)
    plt.hist(X, bins=bins, edgecolor='black')
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(); plt.show()

##### Статистика по датасету
def describe(input_df, p=None, thr=None, null=None, corr=None):
    """
    Описание датасета по ключевым статистикам
    p = перцентиль по которому хотим посмотреть значения и отсечки в клиентах
    thr = аналогично перцентилю, но отсечка в виде абсолютного числа
    null_vals = значения которые будем автоматически обрабатывать как null (например '', np.nan итд); arr
    return df_describe - датасет с описательными статистиками
    const, null, corr - наличие константных признаков, с нуллами, имеющих corr=1 с другими
    """
    df = input_df.copy()
    if null is not None:
        for j in null:
            df = df.replace(j, np.nan)
    dt = df.dtypes.reset_index(name='data_type')
    dtc = df.describe().T[['mean', 'min', 'max', '50%', 'std']].rename(columns={'mean' : 'AVG', '50%' : 'p50'})
    dt = dt.merge(dtc.reset_index(), on='index', how='left')
    dt['null_cnt'] = len(df) - df.count().values
    if corr is not None:
        cr = df.corr(method='spearman') # быстрее
        cr = (cr[cr == 1].sum() > 1).astype(int).reset_index(name='corr_')
        dt = dt.merge(cr, on='index', how='left')
    if p is not None:
        df_pc = df[dtc.index].apply(lambda x: np.percentile(x, p)).reset_index(name=f'p{p}')
        dt = dt.merge(df_pc, how='left')
        outl = pd.DataFrame(None, columns=['index', f'>p{p}_cnt']); j=0
        for col in df_pc['index'].values:
            thr_ = df_pc.loc[df_pc['index'] == col, f'p{p}'].iloc[0]
            outl.loc[j, :] = col, np.sum(df[col] > thr_); j+=1
        dt = dt.merge(outl, how='left')
    if thr is not None:
        outl = pd.DataFrame(None, columns=['index', f'>{thr}_cnt']); j=0
        for col in dtc.index:
            outl.loc[j, :] = col, np.sum(df[col] > thr); j+=1
        dt = dt.merge(outl, how='left')

    dt['uniq_cnt'] = df.nunique().values
    dt['const'] = (dt['std'] == 0).astype(int)

    dt = dt.rename(columns={'index' : 'feature'})
    if input_df.shape[1] != len(dt):
        return 'size error'

    c_list = ['null_cnt', 'const']
    total_result = {}
    total_result['total_dupl_cnt'] = len(df) - df.drop_duplicates().shape[0]
    total_result['const'] = dt['const'].sum()
    total_result['total_cnt'] = len(df)
    total_result['features_cnt'] = len(dt)
    if corr is not None:
        total_result['corr'] = dt['corr_'].sum()
        c_list.append('corr_')
    total_result['null_cnt'] = dt['null_cnt'].sum()
    drop_list = ['std']
    for c in c_list:
        if total_result[c] == 0:
            drop_list.append(c)
    print(total_result)
    return dt.drop(columns=drop_list)
