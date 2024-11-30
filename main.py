# %%
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
import scikitplot as skplt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from lazypredict.Supervised import LazyClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# %% [markdown]
# ## Preprocessing Dat

# %% [markdown]
# ### Načtení dat, info a describe

# %%
data = pd.read_csv("data_titanic.csv", index_col="PassengerId")
data

# %%
data.info()

# %%
data.describe()

# %%
data.describe(include="object")

# %% [markdown]
# ### Chybějící hodnoty a úprava slopců

# %%
# golobální průměr věk
data["Age"].mean()

# %%
# jaký věk doplnit? -> místo např. globálního průměru vezmeme medián podle třídy pasažéra, u kterého chybí hodnota -> přesnější
tabulka_dopln_vek = data[["Age", "Pclass"]].groupby("Pclass").median()
tabulka_dopln_vek

# %%
# řádek po řádku, pokud chybí věk -> je nan, tak najdi jeho třídu a podívej se do tabulky s mediány a doplň, pokud ne nech tam původní věk
data["Age"] = data.apply(lambda row: tabulka_dopln_vek.loc[row["Pclass"]].values[0] if np.isnan(row["Age"]) else row["Age"], axis=1)

# %%
# pokud měl sourozence nebo partnera na palubě -> 1, pokud ne tak 0
data["SibSp"] = data["SibSp"].apply(lambda x: 0 if x == 0 else 1)

# %%
# pokud měl dítě nebo rodiče na palubě -> 1, pokud ne tak 0
data["Parch"] = data["Parch"].apply(lambda x: 0 if x == 0 else 1)

# %%
# Skoro samé unikátní hodnoty, navíc u Cabin drtivá většina chybí
sloupce_ke_smazani = ["Name", "Ticket", "Cabin"]
data = data.drop(columns=sloupce_ke_smazani)

# %%
nejcastejsi_pristav = data["Embarked"].value_counts().idxmax()
data["Embarked"] = data["Embarked"].fillna(nejcastejsi_pristav)

# %% [markdown]
# ### Úprava datovýách typů

# %%
# přetypuj u věku z float na int -> věk se udává v celých číslech
data["Age"] = data["Age"].astype(int)

# %%
# enkódování -> ordinal -> pro každou unikátní hodnotu přiřadíme číslo
pohlavi_na_text = {"female": 0, "male": 1}
data["Sex"] = data["Sex"].apply(lambda x: pohlavi_na_text[x])

# %%
pristavy = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
data["Embarked"] = data["Embarked"].apply(lambda x: pristavy[x])

# %%
data["Embarked"].value_counts()

# %%
embarked_table = pd.get_dummies(data["Embarked"], drop_first=True).astype(int)
embarked_table

# %%
data = data.join(embarked_table).drop(columns=["Embarked"])

# %% [markdown]
# ### Škálování hodnot

# %%
ss_scaler = MinMaxScaler()
data = pd.DataFrame(ss_scaler.fit_transform(data), columns=data.columns)

# %% [markdown]
# ### Korelace a výběr featur

# %%
corr = data.corr()
plt.figure(figsize=(11,8))
sns.heatmap(corr, cmap="Greens",annot=True)
plt.show()

# %%
# treshold, od kdy bereme sloupec jako relevantní prediktor -> vezmeme pro trénink
corr_treshold = 0.2
# vygenereovat list True -> splňuje nebo False -> nesplňuje
splnuje_treshold = list(abs(corr.iloc[0]) >= corr_treshold)
# sloupce tabulky
sloupce = list(data.columns)
# vyber mi jenom slopupce splňující treshold
sloupce_nad_treshold = [sloupce[i] for i in range(len(sloupce)) if splnuje_treshold[i]]

# %%
REDUKOVAT = True
# přepínač, jestli zredukovat sloupce podle korelace nebo ne
if REDUKOVAT:
    data = data[sloupce_nad_treshold]

# %% [markdown]
# ### Rozdělení na trénovací a testovací data

# %%
# rozdělení target a a featury
y = data["Survived"]
X = data.drop(columns=["Survived"])

# %%
# díky stratify rozděl v rámci každé tříty 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# %% [markdown]
# ## Modelování

# %% [markdown]
# ### Lazy predict -> vyzkoušet ve smyčcevšechny sklearn modely se základním nastavením a jednoduchým preprocessingem

# %%
lc = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = lc.fit(X_train, X_test, y_train, y_test)

models

# %% [markdown]
# ### Vybrali jsme model podle Lazy Predict -> hledání optimálních parametrů pomocí grid search

# %%
# knn podle lazy predict vypadá slibně
knn = KNeighborsClassifier()
# parametry, podle kterých budu hledat optimální nastavení modelu
parametry = {'metric':('minkowski', 'manhattan', 'cosine'), 'weights':('uniform', 'distance'), 'n_neighbors':[3, 5, 7, 9]}
# inicializace grid search -> optimalizace - hledání nejlepší kombinace pro daný model a parametry
gs = GridSearchCV(knn, parametry)

# %%
# provedení hodně tréninků pomocí k-fold a najití nejlepší kombinace parametrů
gs.fit(X_train, y_train)

# %%
# vítězné parametry
gs.best_params_

# %%
# model s nejlepšími parametry
best_knn = gs.best_estimator_

# %%
# predikce pomocí nejlepšího modelu
y_best_knn = best_knn.predict(X_test)

# %% [markdown]
# ### Vyhodnocení na test datech

# %%
print(classification_report(y_test, y_best_knn))

# %%
# Confusion matice
cm = confusion_matrix(y_test, y_best_knn, labels=best_knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=best_knn.classes_)
disp.plot()
plt.show()

# %%
# výstup modelu jako číslo -> ne třída (.predict_proba vs .predict)
y_best_knn_proba = best_knn.predict_proba(X_test)

# nakreslit ROC křivku
skplt.metrics.plot_roc_curve(y_test, y_best_knn_proba)
plt.show()


