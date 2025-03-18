import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import graphviz
import xgboost as xgb

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler


df = pd.read_excel("makine öğrenmesi/makine.xlsx")

print(df.value_counts())

print(df.isnull().sum())

df = df.drop("deney no", axis=1)

y = df['sertlik']
x=df.drop('sertlik',axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#standartlaştırma
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#normalizsyon
mm = MinMaxScaler()
x_minmax = mm.fit_transform(x)
x_minmax = pd.DataFrame(x_minmax, columns=x.columns)
print(x_minmax.head(3))
print(f"Maksimum Değeri: {x_minmax.max().max()}")
print(f"Minimum Değeri: {x_minmax.min().min()}")


correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns)
plt.title('Korelasyon Matrisi')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Lineer Regresyon Modeli
lr = LinearRegression()
model_lr = lr.fit(x_train, y_train)

# Lasso ve Ridge Regresyon Modelleri
model_lasso = Lasso(alpha=0.1).fit(x_train, y_train)
model_ridge = Ridge(alpha=0.1).fit(x_train, y_train)

# Karar Ağacı Modeli
tree = DecisionTreeRegressor()
model_tree = tree.fit(x_train, y_train)

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=200)
model_rf = rf.fit(x_train, y_train)

# XGBoost Regressor
xg = xgb.XGBRegressor()
model_xg = xg.fit(x_train, y_train)

# SVR Modeli
svr = SVR(kernel='rbf')
model_svr = svr.fit(x_train_scaled, y_train)

# Gradient Boosting Regressor Modeli
gb_model = GradientBoostingRegressor()
gb_model.fit(x_train_scaled, y_train)
y_pred_gb = gb_model.predict(x_test_scaled)


print(f"Lineer Regresyon Test Skoru: {model_lr.score(x_test, y_test)}")
print(f"Karar Ağacı Test Skoru: {model_tree.score(x_test, y_test)}")
print(f"Random Forest Test Skoru: {model_rf.score(x_test, y_test)}")
print(f"XGBoost Test Skoru: {model_xg.score(x_test, y_test)}")
print(f"SVR Test Skoru: {model_svr.score(x_test_scaled, y_test)}")
print("Gradient Boosting Test Seti Skoru:", r2_score(y_test, y_pred_gb))

# RandomForest Modelini çapraz dogrulama ile değerlendirme

rf_model = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=5, min_samples_leaf=1)
cv_scores = cross_val_score(rf_model, x, y, cv=10, scoring='r2')
print("Cross Validation Skorları:", cv_scores)
print("Ortalama Cross Validation Skoru:", cv_scores.mean())

# RandomizedSearchCV ile Random Forest için hiperparametre optimizasyonu
param_dist_rf = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6]
}

random_search_rf = RandomizedSearchCV(
    estimator=RandomForestRegressor(),
    param_distributions=param_dist_rf,
    n_iter=100,
    cv=5,
    scoring='r2',
    random_state=42
)
random_search_rf.fit(x_train_scaled, y_train)
print("En İyi RandomForest Parametreler (RandomizedSearchCV):", random_search_rf.best_params_)
print("En İyi RandomForest Skor (RandomizedSearchCV):", random_search_rf.best_score_)

# En iyi modeli kullanarak test setinde tahmin yapma

best_rf_model = random_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(x_test_scaled)
print("RandomForest Test Seti Skoru (RandomizedSearchCV):", r2_score(y_test, y_pred_rf))

# # XGBoost modeli

# xgb_model = xgb.XGBRegressor()
# xgb_model.fit(x_train_scaled, y_train)
# y_pred_xgb = xgb_model.predict(x_test_scaled)
#print("XGBoost Test Seti Skoru:", r2_score(y_test, y_pred_xgb))

# Tahminleri görselleştirme

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Random Forest Tahmin Edilen Değerler')
plt.title('Random Forest - Gerçek vs. Tahmin')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_gb, alpha=0.5)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Gradient Boosting Tahmin Edilen Değerler')
plt.title('Gradient Boosting - Gerçek vs. Tahmin')

plt.show()

# # XGBoost ağacını görselleştirme

# xgb.plot_tree(model_xg, num_trees=0)
# plt.rcParams['figure.figsize'] = [15, 5]
# plt.show()
# plt.savefig('xgb_tree.png', dpi=300)

# Karar Ağacı ağacını görselleştirme
dot = export_graphviz(model_tree, feature_names=x.columns, filled=True)
gorsel = graphviz.Source(dot)
gorsel.render('tree')
gorsel.view()

# Random Forest ağaçlarından birini görselleştirme
estimator = model_rf.estimators_[0]
dot_data = export_graphviz(estimator, feature_names=x.columns, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("random_forest_tree")
graph.view()

# Hatalar DataFrame oluşturma ve kaydetme
df_hata = pd.DataFrame()
df_hata["gerçek"] = y
y_tahmin = model_lr.predict(x)
df_hata["tahmin"] = y_tahmin
df_hata["hata"] = y - y_tahmin
df_hata.to_excel("hatalar.xlsx", index=False)

