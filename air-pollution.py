# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd

data = pd.read_csv("data/air_pollution_data.csv")

# %%
data.head()

# %%
data

# %%
data.info()

# %%
data.describe()

# %%
data.isna().sum()

# %%
data.nunique()

# %%
data["city_name"].unique()

# %%
data = data.drop_duplicates()

# %%
data = data.dropna(subset="country_name")

# %%
data.isna().sum()

# %%
import matplotlib_inline
from matplotlib import pyplot as plt

# %%
data["aqi_value"].unique()


# %%
def create_hist(data: list[float], bins: int) -> None:
    plt.rcParams.update({"font.size": 8})
    plt.figure(figsize=(4, 3))
    plt.hist(data, bins=bins)
    plt.show()


# %%
create_hist(data["aqi_value"], 50)

# %%
data.info()

# %%
data.dtypes

# %%
data["city_name"] = data["city_name"].astype(pd.StringDtype())

# %%
data.loc[data["country_name"] == "Germany"]["city_name"].unique()

# %%
plt.scatter(data["aqi_value"], data["pm2.5_aqi_value"])

# %% [markdown]
# # Próba zastosowania geodatasets i geopandas do utworzenia mapki
#
# # %%
# import geodatasets
# import geopandas as gpd
#
# g_data = gpd.read_file("data/world-administrative-boundaries.zip")
# locations = gpd.tools.geocode(g_data["name"])
#
# # %%
# locations
#
# # %%
# fig, ax = plt.subplots()
# g_data.to_crs("EPSG:4326").plot(ax=ax, color="white", edgecolor="black")
# locations.plot(ax=ax, color="red")

# %% [markdown]
# # Zbudowanie średniej wartości aqi_value dla poszczególnych państw

# %%
from collections import defaultdict

import numpy as np

values_per_country = defaultdict(list)
for country, aqi in zip(data["country_name"], data["aqi_value"]):
    values_per_country[country].append(aqi)

avg_per_country = {country: np.mean(values) for country, values in values_per_country.items()}

# %% [markdown]
# ## Dane w dataframe posortowane wg. aqi_value ASC

# %%
avg_per_country_df = (
    pd.DataFrame([avg_per_country.keys(), avg_per_country.values()])
    .transpose()
    .rename(columns={0: "country_name", 1: "aqi_value"})
    .sort_values(by="aqi_value", ascending=False)
    .reset_index(drop=True)
)
avg_per_country_df.head(n=15)

# %% [markdown]
# ## Barplot top 15 średnich

# %%
# %matplotlib inline

import seaborn as sns

top_15 = avg_per_country_df[:15]
plt.rcParams["figure.figsize"] = [10, 5]

plt.xticks(rotation=75)
sns.barplot(
    x=top_15["country_name"],
    y=top_15["aqi_value"],
    palette=sns.cubehelix_palette(len(top_15)),
    hue=top_15["aqi_value"],
    legend=False,
)
plt.show()

# # %% [markdown]
# # ## Mapka średnich wartości aqi_value per państwo
# #
# # Dla państw bez danych przyjęto wartość 0, która realnie nie jest osiągalna.
#
# # %%
# g_data["aqi_values"] = pd.Series(
#     [avg_per_country.get(country_name, 0) for country_name in g_data["name"]]
# )
# g_data.explore(column="aqi_values")

# %% [markdown]
# ## Próba dedukcji wysokiego poziomu aqi_value w Korei Południowej

# %%
values_per_country["Republic of Korea"]

# %%
data[data["country_name"] == "Republic of Korea"]

# %%
data[data["country_name"] == "Poland"].max()

# %% [markdown]
# ## Model pomysły
#
# - estymacja jednego parametru pod wpływem innych parametrów jakości powietrza
# - estymacja wielu parametrów jakości powietrza pod wpływem jednego/wielu
# - propozycje lokalizacji pod wpływem parametrów
# - kategoria aqi_category w zależności od parametrów
#
# Wybrany model: estymacja jednego parametru pod wpływem innych parametrów jakości powietrza
#
# ## Klasteryzacja
#
# t-SNE zrobić po subsetcie fit, a po całości transform przynajmniej jedna osoba powinna zrobić
#
# ##
#

# %% [markdown]
# ## Korelacje

# %%
for_corr = data[
    [
        "aqi_value",
        "co_aqi_value\t",
        "ozone_aqi_value",
        "no2_aqi_value",
        "pm2.5_aqi_value",
    ]
]
corr = for_corr.corr()
corr.style.background_gradient(cmap="coolwarm")

# %% [markdown]
# ## Przygotowanie do tworzenia modelu

# %%
X = np.array(
    [
        data["co_aqi_value\t"],
        data["ozone_aqi_value"],
        data["no2_aqi_value"],
        data["pm2.5_aqi_value"],
    ]
).T
Y = np.array(data["aqi_value"]).T

# %% [markdown]
# ## TSNE

# %%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# tsne = TSNE(n_components=2, learning_rate="auto", init="random")
#
# # %%
# X_embedded = tsne.fit_transform(X)
# X_embedded.shape
#
# # %%
# X_embedded
#
# # %%
# plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
# plt.show()
#
# # %%
# tsne = TSNE(n_components=3, learning_rate="auto", init="random")
#
# X3_embedded = tsne.fit_transform(X)
# X3_embedded.shape
#
# # %%
# # %matplotlib widget
#
# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# ax.scatter(X3_embedded[:, 0], X3_embedded[:, 1], X3_embedded[:, 2], s=3)
# plt.show()
#
# # %%
# import plotly
# import plotly.graph_objs as go
#
# plotly.offline.init_notebook_mode()
#
# trace = go.Scatter3d(
#     x=X3_embedded[:, 0],
#     y=X3_embedded[:, 1],
#     z=X3_embedded[:, 2],
#     mode="markers",
#     marker={
#         "size": 10,
#         "opacity": 0.8,
#     },
# )
#
# layout = go.Layout(margin={"l": 0, "r": 0, "b": 0, "t": 0})
#
# _data = [trace]
# plot_figure = go.Figure(data=_data, layout=layout)
#
# plotly.offline.iplot(plot_figure)
#
# %% [markdown]
# ## PCA

# %%
pca = PCA(n_components="mle")
pca.fit(X)

# %%
pca.explained_variance_ratio_

# %%
pca.explained_variance_

# %%
pca.singular_values_

# %%
pca.components_

# %%
X_fitted = pca.fit_transform(X)

X_fitted

# %%
X_fitted.shape

# %%
# %matplotlib widget

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=3)
plt.show()

# %% [markdown]
# # Rozdział na zbiór treningowy i testowy

# %%
from sklearn.model_selection import train_test_split

random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, Y.T, test_size=0.2, random_state=42)

# %%
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors=5)
knr.fit(X_train, y_train)

# %%
from sklearn.metrics import mean_squared_error

y_pred = knr.predict(X_test)
mean_squared_error(y_test, y_pred)

# %%
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)

# %%
y_test

# %%
y_pred

# %%
differences = np.argmax([pred - test for pred, test in zip(y_pred, y_test)])
differences

# %%
y_pred[2241]

# %%
X_test[2241]

# %%
data[data["pm2.5_aqi_value"] == 319]

# %%
data[data["aqi_category"] == "Hazardous"][data["country_name"] == "India"]

# %%
from sklearn.tree import DecisionTreeRegressor

decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, y_train)

# %%
y_pred = decision_tree.predict(X_test)
mean_squared_error(y_test, y_pred)

# %%
r2_score(y_test, y_pred)

# %%
differences = np.argmax([pred - test for pred, test in zip(y_pred, y_test)])
differences


# %%
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=300)
lasso.fit(X_train, y_train)

# %%
y_pred = lasso.predict(X_test)
mean_squared_error(y_test, y_pred)

# %%
r2_score(y_test, y_pred)

# %%
differences = np.argmax([pred - test for pred, test in zip(y_pred, y_test)])
differences

# %%
from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor()
random_forest.fit(X_train, y_train)

# %%
y_pred = random_forest.predict(X_test)
mean_squared_error(y_test, y_pred)

# %%
r2_score(y_test, y_pred)

# %%
differences = np.argmax([pred - test for pred, test in zip(y_pred, y_test)])
differences

# %%
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(random_state=random_state, max_iter=500)
mlp.fit(X_train, y_train)

# %%

y_pred = mlp.predict(X_test)
mean_squared_error(y_test, y_pred)

# %%
r2_score(y_test, y_pred)

# %%
differences = np.argmax([pred - test for pred, test in zip(y_pred, y_test)])
differences

# %% [markdown]
# ## Walidacja krzyżowa

# %%
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# %% [markdown]
#
# #### k-neighbours regressor model

# %%
from sklearn.neighbors import KNeighborsRegressor
knr_metrics = {
    "mean_squared_error": [],
    "r2_score": []
}
for i, (train_index, test_index) in enumerate(kf.split(X, Y.T)):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # Trenowanie modelu
    knr = KNeighborsRegressor(n_neighbors=5)
    knr.fit(X_train, Y_train)

    # Testy modelu
    knr_Y_pred = knr.predict(X_test)
    
    knr_mean_squared_error = mean_squared_error(Y_test, knr_Y_pred)
    knr_r2_score = r2_score(Y_test, knr_Y_pred)

    knr_metrics["mean_squared_error"].append(knr_mean_squared_error)
    knr_metrics["r2_score"].append(knr_r2_score)
    print("[ FOLD", i, "]")
    print("    Mean squared error:", knr_mean_squared_error)
    print("    R2 score:", knr_r2_score)

# %% [markdown]
# #### Decision tree model

# %%
from sklearn.tree import DecisionTreeRegressor
tree_metrics = {
    "mean_squared_error": [],
    "r2_score": []
}
for i, (train_index, test_index) in enumerate(kf.split(X, Y.T)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, Y_test = Y[train_index], Y[test_index]

    # Trenowanie modelu
    decision_tree = DecisionTreeRegressor()
    decision_tree.fit(X_train, y_train)

    # Testy modelu
    tree_Y_pred = decision_tree.predict(X_test)

    tree_mean_squared_error = mean_squared_error(Y_test, tree_Y_pred)
    tree_r2_score = r2_score(Y_test, tree_Y_pred)

    tree_metrics["mean_squared_error"].append(tree_mean_squared_error)
    tree_metrics["r2_score"].append(tree_r2_score)
    print("[ FOLD", i, "]")
    print("    Mean squared error:", tree_mean_squared_error)
    print("    R2 score:", tree_r2_score)


# %% [markdown]
# #### Lasso model

# %%
from sklearn.linear_model import Lasso
lasso_metrics = {
    "mean_squared_error": [],
    "r2_score": []
}
for i, (train_index, test_index) in enumerate(kf.split(X, Y.T)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, Y_test = Y[train_index], Y[test_index]

    # Trenowanie modelu
    lasso = Lasso(alpha=300)
    lasso.fit(X_train, y_train)

    # Testy modelu
    lasso_Y_pred = decision_tree.predict(X_test)

    lasso_mean_squared_error = mean_squared_error(Y_test, lasso_Y_pred)
    lasso_r2_score = r2_score(Y_test, lasso_Y_pred)

    lasso_metrics["mean_squared_error"].append(lasso_mean_squared_error)
    lasso_metrics["r2_score"].append(lasso_r2_score)
    print("[ FOLD", i, "]")
    print("    Mean squared error:", lasso_mean_squared_error)
    print("    R2 score:", lasso_r2_score)


# %% [markdown]
# #### Random forest model

# %%
from sklearn.ensemble import RandomForestRegressor
forest_metrics = {
    "mean_squared_error": [],
    "r2_score": []
}
for i, (train_index, test_index) in enumerate(kf.split(X, Y.T)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, Y_test = Y[train_index], Y[test_index]

    # Trenowanie modelu
    forest = RandomForestRegressor()
    forest.fit(X_train, y_train)

    # Testy modelu
    forest_Y_pred = decision_tree.predict(X_test)

    forest_mean_squared_error = mean_squared_error(Y_test, forest_Y_pred)
    forest_r2_score = r2_score(Y_test, forest_Y_pred)

    forest_metrics["mean_squared_error"].append(forest_mean_squared_error)
    forest_metrics["r2_score"].append(forest_r2_score)
    print("[ FOLD", i, "]")
    print("    Mean squared error:", forest_mean_squared_error)
    print("    R2 score:", forest_r2_score)


# %% [markdown]
# #### MPL Regressor

# %%
from sklearn.neural_network import MLPRegressor
mlp_mean_squared_error = []
mlp_r2_score = []
for train_index, test_index in kf.split(X, Y.T):
    X_train, X_test = X[train_index], X[test_index]
    y_train, Y_test = Y[train_index], Y[test_index]

    # Trenowanie modelu
    mlp = MLPRegressor(random_state=random_state, max_iter=500)
    mlp.fit(X_train, y_train)

    # Testy modelu
    mlp_Y_pred = decision_tree.predict(X_test)
    mlp_mean_squared_error.append(mean_squared_error(Y_test, mlp_Y_pred))
    mlp_r2_score.append(r2_score(Y_test, mlp_Y_pred))

# %%
