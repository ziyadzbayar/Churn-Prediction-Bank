# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 16:20:14 2026

@author: ziyad
"""

# -*- coding: utf-8 -*-
"""
Mini-Projet Churn - Comparaison de Modèles
Régression Logistique - Arbre de Décision - XGBoost

Auteur : Ziad Zbayar
Date : Janvier 2026
"""

# =============================
# 1. Importations
# =============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# =============================
# 2. Chargement des données
# =============================

df = pd.read_csv(r"C:\MiniProjet\Churn_Modelling.csv")

print("Aperçu des données :")
print(df.head())

print("\nValeurs manquantes :")
print(df.isnull().sum())
# ================================================
# . Visualisation variable cible et autre variable
# ================================================

plt.figure()
df['Exited'].value_counts().plot(kind='bar')
plt.title("Distribution de la variable cible (Exited)")
plt.xlabel("Classe (0 = Reste, 1 = Churn)")
plt.ylabel("Nombre de clients")
plt.show()

features_to_plot = [
    'CreditScore', 'Age', 'Balance', 'EstimatedSalary',
    'NumOfProducts', 'Tenure'
]

for col in features_to_plot:
    plt.figure()
    df[col].hist(bins=30)
    plt.title(f"Histogramme de {col}")
    plt.xlabel(col)
    plt.ylabel("Fréquence")
    plt.show()

# =============================
# 3. Prétraitement
# =============================

# Encodage
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

# Suppression colonnes inutiles
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# =============================
# 5. Matrice de corrélation (Heatmap)
# =============================

corr_matrix = df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title("Heatmap de corrélation des variables")
plt.show()

print("Correlation matricielle :")
print(corr_matrix)

# Features & Target
X = df.drop('Exited', axis=1)
y = df['Exited']


# =============================
# 4. Split Train / Test
# =============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# 5. Standardisation (pour Logistic)
# =============================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================
# 6. Régression Logistique
# =============================

log_model = LogisticRegression(
    max_iter=5000,
    class_weight='balanced',
    solver='lbfgs'
)

log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)
y_proba_log = log_model.predict_proba(X_test_scaled)[:, 1]

# =============================
# 7. Arbre de Décision
# =============================

dt_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_split=10,
    random_state=42
)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
y_proba_dt = dt_model.predict_proba(X_test)[:, 1]

# =============================
# 8. XGBoost
# =============================


xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    reg_alpha=1,
    reg_lambda=2,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# =============================
# 9. Fonction d'évaluation
# =============================

def evaluate_model(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "="*40)
    print(f" Modèle : {name}")
    print("="*40)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"AUC       : {auc:.4f}")
    print("\nMatrice de confusion :")
    print(cm)
    print("\nRapport de classification :")
    print(classification_report(y_true, y_pred))


# =============================
# 10. Résultats finaux
# =============================

evaluate_model("Régression Logistique", y_test, y_pred_log, y_proba_log)
evaluate_model("Arbre de Décision", y_test, y_pred_dt, y_proba_dt)
evaluate_model("XGBoost", y_test, y_pred_xgb, y_proba_xgb)

# =============================
# 11. Conclusion automatique simple
# =============================

print("\nComparaison terminée.")
print("Utiliser les métriques (Recall, F1, AUC) pour choisir le meilleur modèle.")

# =============================
# 12. Exemples de prédictions XGBoost
# =============================

results_xgb = X_test.copy()
results_xgb['Churn_Reel'] = y_test.values
results_xgb['Churn_Predit_XGB'] = y_pred_xgb
results_xgb['Proba_Churn_XGB'] = y_proba_xgb

print("\nExemples de prédictions XGBoost (10 premiers clients) :")
print(results_xgb[['Churn_Reel', 'Churn_Predit_XGB', 'Proba_Churn_XGB']].head(10))
