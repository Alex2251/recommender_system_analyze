import pandas as pd
import numpy as np


def RMSE(df_predict, df_real):
    substraction = df_predict - df_real
    rmse = (substraction ** 2).mean() ** 0.5
    return rmse


def MAE(df_predict, df_real):
    substraction = df_predict - df_real
    mae = substraction.mean()
    return mae


def map_5(df_predict, users, mask, ntop=5):
    avg_map_5 = {}
    for usr in users:
        df_user = df_predict[usr].sort_values(ascending=False)
        df_user_valid = (df_user*mask[usr].loc[df_user.index]).iloc[:ntop]
        index_relevant_rating = df_user_valid.reset_index()
        index_relevant_rating = index_relevant_rating[index_relevant_rating[usr] > 3].index
        enumerator = 0
        avg_pr_tmp = []
        for i, rnk in enumerate(index_relevant_rating):
            enumerator += 1
            avg_pr_tmp += [enumerator/(rnk+1)]

        if (len(avg_pr_tmp) != 0):
            avg_map_5[usr] = np.mean(avg_pr_tmp)
        else:
            avg_map_5[usr] = 0

    return avg_map_5


def precision_recall(df_predict, df_real,  users, mask, ntop=5, count=3):
    pr = {}
    rec = {}
    for usr in users:
        df_user = df_predict[usr].sort_values(ascending=False)
        df_user_valid = (df_user * mask[usr].loc[df_user.index]).iloc[:ntop]
        index = [df_user_valid.index[i] for i in range(ntop) if df_user_valid.iloc[i] > count]
        row_ratings = (df_real[usr].loc[index] > count).astype('int')
        fil_ratings = (df_user_valid.loc[index] > count).astype('int')
        enumerator = (row_ratings.values == fil_ratings.values).sum()
        pr[usr] = enumerator / ntop
        rec[usr] = enumerator / (df_real[usr] > count).astype('int').sum()
    return pr, rec


def diversity_among_price(df_predict, df_items, users, ntop=5):
    div = {}
    for usr in users:
        indexes = df_predict[usr].sort_values(ascending=False)[:ntop].index
        prices = df_items.loc[indexes]['Price']
        norm = list(map(lambda x: (x - prices.min()) / (prices.max() - prices.min()), prices))
        div[usr] = np.std(norm)
    return div


def diversity_among_category(df_predict, df_items, users, ntop=5):
    def coef_diversity(df_cat):
        diff = []
        for i, k in enumerate(df_cat.columns[:-1]):
            df_diff = df_cat[df_cat.columns[i + 1:]].apply(lambda x: x != df_cat[k])
            diff += df_diff.apply(lambda x: diff_level[x[x].index[0]] if len(x[x]) > 0 else 0).tolist()
        return np.mean(diff)

    dict_level = {'Diff Cats.': 0, 'Diff Subcats.': 1, 'Diff Subsubcats': 2, 'Diff Leafcat': 3}
    diff_level = {0: 1, 1: 0.75, 2: 0.5, 3: 0.25}

    diversity = {}
    for usr in users:
        items = df_predict[usr].sort_values(ascending=False)[:ntop].index
        df_items_usr = df_items.loc[items]
        categories = df_items_usr['FullCat'].apply(lambda x: x.split('/')[1:])
        df_categories = pd.DataFrame(index=dict_level.values())
        for i in df_items_usr.index:
            df_categories = df_categories.join(pd.DataFrame(categories.loc[i], columns=[str(i)]))
        diversity[usr] = coef_diversity(df_categories)
    return pd.Series(diversity).mean()


def personalization(df_predict, users, ntop=5):
    def difference_k(list1, list2):
        diff_elements = [i for i in list2 if i not in list1]
        return len(diff_elements) / len(list1)

    sim = []
    for num, usr in enumerate(users[:-1]):
        indexes = df_predict[usr].sort_values(ascending=False)[:ntop].index
        sim += [difference_k(indexes, df_predict[i].sort_values(ascending=False)[:ntop].index) for i in users[num + 1:]]
    return np.mean(sim)


def coverage(df_predict, users, ntop=5):
    cov = []
    for usr in users:
        indexes = df_predict[usr].sort_values(ascending=False)[:ntop].index
        cov += list(indexes)
    return len(set(cov))


def serendipity(df_predict, users, mask, ntop=5):
    serend = []
    for usr in users:
        data = df_predict[usr].sort_values(ascending=False)[:ntop]
        validity = (data > 3).astype('int')
        prim = 5 * mask.loc[data.index].sum(axis=1) / mask.shape[1]
        serend += [((data - prim) * validity).sum() / ntop]
    return np.mean(serend)


def mean_availability(df_predict, df_items, users, ntop=5):
    av_users = {}
    for usr in users:
        data = df_predict[usr].sort_values(ascending=False)[:ntop]
        data_availability = df_items.loc[data.index]['Availability'].mean()
        av_users[usr] = data_availability
    return pd.Series(av_users).mean()


def reset_top_ratings_by_price(df_predict, df_items, users, ntop=5):

    def calc_price_diff(data):
        price = df_items.loc[data.index, 'Price']
        price = price.apply(lambda x: (x-min(price))/(max(price)-min(price)))
        dif_price = price.apply(lambda x: x - price.iloc[0])
        sorted_values = data + 5 * dif_price
        sorted_values = sorted_values.sort_values(ascending=False)[:ntop]
        return sorted_values

    data_locl = df_predict.copy()
    for usr in users:
        data = data_locl[usr].sort_values(ascending=False)[:2*ntop]
        new_ratings_top_data = calc_price_diff(data)
        data_locl.loc[new_ratings_top_data.index, usr] = new_ratings_top_data
    return data_locl


def reset_top_ratings_by_category(df_predict, df_items, users, ntop=5):
    def coef_diversity(df_cat):
        comparison_cat = df_cat.apply(
            lambda x: x[x != df_cat[df_cat.columns[0]]].index[0] \
                if len(x[x != df_cat[df_cat.columns[0]]].index) > 0 else -1
        )
        return comparison_cat.apply(lambda x: diff_level[x])

    def calc_cat_diff(data):
        df_user_items = df_items.loc[data.index, 'FullCat'].apply(lambda x: x.split('/')[1:])
        df_categories = pd.DataFrame(index=dict_level.values())
        for i in df_user_items.index:
            df_categories = df_categories.join(pd.DataFrame(df_user_items.loc[i], columns=[i]))
        sorted_values = data + 5 * coef_diversity(df_categories)
        sorted_values = sorted_values.sort_values(ascending=False)[:ntop]
        return sorted_values

    dict_level = {'Diff Cats.': 0, 'Diff Subcats.': 1, 'Diff Subsubcats': 2, 'Diff Leafcat': 3}
    diff_level = {-1: 0, 0: 1, 1: 0.75, 2: 0.5, 3: 0.25}

    data_locl = df_predict.copy()
    for usr in users:
        data = data_locl[usr].sort_values(ascending=False)[:2 * ntop]
        new_ratings_top_data = calc_cat_diff(data)
        data_locl.loc[new_ratings_top_data.index, usr] = new_ratings_top_data
    return data_locl