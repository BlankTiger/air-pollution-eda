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

# %%
import geodatasets
import geopandas as gpd

g_data = gpd.read_file("data/world-administrative-boundaries.zip")
locations = gpd.tools.geocode(g_data["name"])

# %%
locations

# %%
fig, ax = plt.subplots()
g_data.to_crs("EPSG:4326").plot(ax=ax, color="white", edgecolor="black")
locations.plot(ax=ax, color="red")

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

# %% [markdown]
# ## Mapka średnich wartości aqi_value per państwo
#
# Dla państw bez danych przyjęto wartość 0, która realnie nie jest osiągalna.

# %%
g_data["aqi_values"] = pd.Series(
    [avg_per_country.get(country_name, 0) for country_name in g_data["name"]]
)
g_data.explore(column="aqi_values")

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

tsne = TSNE(n_components=2, learning_rate="auto", init="random")

# %%
X_embedded = tsne.fit_transform(X)
X_embedded.shape

# %%
X_embedded

# %%
plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
plt.show()

# %%
tsne = TSNE(n_components=3, learning_rate="auto", init="random")

X3_embedded = tsne.fit_transform(X)
X3_embedded.shape

# %%
# %matplotlib widget

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(X3_embedded[:, 0], X3_embedded[:, 1], X3_embedded[:, 2], s=3)
plt.show()

# %%
import plotly
import plotly.graph_objs as go

plotly.offline.init_notebook_mode()

trace = go.Scatter3d(
    x=X3_embedded[:, 0],
    y=X3_embedded[:, 1],
    z=X3_embedded[:, 2],
    mode="markers",
    marker={
        "size": 10,
        "opacity": 0.8,
    },
)

layout = go.Layout(margin={"l": 0, "r": 0, "b": 0, "t": 0})

_data = [trace]
plot_figure = go.Figure(data=_data, layout=layout)

plotly.offline.iplot(plot_figure)

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
