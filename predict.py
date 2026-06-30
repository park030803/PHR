"""혈압 위험 예측 모델 — SYNTHETIC PHR 데이터 기반"""
import pandas as pd
import numpy as np
import json
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, f1_score, precision_score, recall_score)
from sklearn.inspection import permutation_importance
warnings.filterwarnings('ignore')

# ── 1. Load & Clean ──────────────────────────────────────────
CSV_PATH = r'C:\Users\USER\Desktop\PHR분석실무\SYNTHETIC_PHR_MEDICAL_v4_FINAL_KR_HEADER.csv'
df = pd.read_csv(CSV_PATH, encoding='utf-8-sig', skiprows=1)

# Standardize column names to English
COL_MAP = {
    '만나이': 'AGE', '성별': 'SEX', '흡연상태': 'SMOKING', '음주빈도': 'DRINKING',
    '운동습관': 'EXERCISE', '평균수면시간': 'SLEEP_HOURS', '스트레스수준': 'STRESS_LEVEL',
    '주관적건강상태': 'SUBJECTIVE_HEALTH', '생활습관위험점수': 'LIFESTYLE_RISK_SCORE',
    '체질량지수': 'BMI', '허리둘레': 'WAIST', '수축기혈압': 'SBP',
    '이완기혈압': 'DBP', '공복혈당': 'GLUCOSE', 'HDL콜레스테롤': 'HDL',
    'LDL콜레스테롤': 'LDL', '중성지방': 'TRIGLYCERIDE', '고혈압상태': 'HYPERTENSION',
    '당뇨상태': 'DIABETES', '이상지질혈증상태': 'DYSLIPIDEMIA',
    '대사증후군구성요소수': 'METABOLIC_SYNDROME_COUNT', '대사증후군여부': 'METABOLIC_SYNDROME',
    '가족력': 'FAMILY_HISTORY', '복약정보': 'MEDICATION', '혈압약복용': 'TAKES_ANTIHYPERTENSIVE',
    '당뇨약복용': 'TAKES_DIABETES_MED', '고지혈증약복용': 'TAKES_LIPID_MED',
    '심혈관위험점수': 'CARDIO_RISK_SCORE', '심혈관위험등급': 'CARDIO_RISK',
    '폐경여부': 'MENOPAUSE', 'ASCVD위험도': 'ASCVD_RISK_LEVEL', '심장나이': 'HEART_AGE'
}
df = df.rename(columns=COL_MAP)

# Drop columns that are direct leaks or highly redundant with the target
LEAK_COLS = ['CARDIO_RISK_SCORE', 'CARDIO_RISK', 'ASCVD_RISK_LEVEL', 'HEART_AGE',
             'TAKES_ANTIHYPERTENSIVE', 'TAKES_DIABETES_MED', 'TAKES_LIPID_MED',
             'MEDICATION', 'METABOLIC_SYNDROME_COUNT', 'METABOLIC_SYNDROME',
             'DIABETES', 'DYSLIPIDEMIA', 'FAMILY_HISTORY']
FEATURE_COLS = ['AGE', 'SEX', 'SMOKING', 'DRINKING', 'EXERCISE', 'SLEEP_HOURS',
                'STRESS_LEVEL', 'SUBJECTIVE_HEALTH', 'LIFESTYLE_RISK_SCORE',
                'BMI', 'WAIST', 'SBP', 'DBP', 'GLUCOSE', 'HDL', 'LDL', 'TRIGLYCERIDE',
                'MENOPAUSE']
TARGET_COL = 'HYPERTENSION'

# Drop leak columns
df_feat = df.drop(columns=LEAK_COLS, errors='ignore')

# Convert numeric columns
for col in ['SLEEP_HOURS', 'LIFESTYLE_RISK_SCORE']:
    df_feat[col] = pd.to_numeric(df_feat[col], errors='coerce')

# Drop rows with missing target
df_feat = df_feat.dropna(subset=[TARGET_COL])

# ── 2. Target Simplification (3-class → 2-class: hypertensive vs not) ──
ht_values = df_feat[TARGET_COL].unique()
print(f"고혈압 상태 값: {ht_values}")
print(df_feat[TARGET_COL].value_counts())

# Binary classification: "고혈압" → 1, everything else → 0
df_feat['HTN_BINARY'] = df_feat[TARGET_COL].apply(lambda x: 1 if str(x).strip() == '고혈압' else 0)

# Also keep 3-class for ordinal analysis
htn_map = {'정상': 0, '고혈압 전단계': 1, '고혈압': 2}
df_feat['HTN_3CLASS'] = df_feat[TARGET_COL].map(htn_map)

print(f"\nBinary hypertension: {df_feat['HTN_BINARY'].value_counts().to_dict()}")
print(f"3-Class hypertension: {df_feat['HTN_3CLASS'].value_counts().to_dict()}")

# ── 3. Encode Categorical Features ────────────────────────────
categorical_cols = ['SEX', 'SMOKING', 'DRINKING', 'EXERCISE', 'STRESS_LEVEL',
                    'SUBJECTIVE_HEALTH', 'MENOPAUSE']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_feat[col + '_ENC'] = le.fit_transform(df_feat[col].astype(str))
    encoders[col] = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}

encoded_feature_cols = ['AGE', 'SEX_ENC', 'SMOKING_ENC', 'DRINKING_ENC', 'EXERCISE_ENC',
                        'SLEEP_HOURS', 'STRESS_LEVEL_ENC', 'SUBJECTIVE_HEALTH_ENC',
                        'LIFESTYLE_RISK_SCORE', 'BMI', 'WAIST', 'SBP', 'DBP',
                        'GLUCOSE', 'HDL', 'LDL', 'TRIGLYCERIDE', 'MENOPAUSE_ENC']

# Drop rows with missing features
df_model = df_feat.dropna(subset=encoded_feature_cols + ['HTN_BINARY'])
X = df_model[encoded_feature_cols].values
y = df_model['HTN_BINARY'].values

print(f"\nFull dataset: {len(df)} rows → after cleaning: {len(df_model)} rows")
print(f"Positive class (고혈압): {y.sum()} ({y.mean()*100:.1f}%)")

# ── 4. Train/Test Split + Scale ──────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 5. Train Models ──────────────────────────────────────────
results = {}
models_info = {}

# 5.1 Logistic Regression
lr = LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
lr_cv = cross_val_score(lr, X_train_scaled, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc')
results['LogisticRegression'] = {
    'accuracy': round(accuracy_score(y_test, lr_pred), 4),
    'f1': round(f1_score(y_test, lr_pred), 4),
    'precision': round(precision_score(y_test, lr_pred), 4),
    'recall': round(recall_score(y_test, lr_pred), 4),
    'roc_auc': round(roc_auc_score(y_test, lr_proba), 4),
    'cv_roc_auc_mean': round(lr_cv.mean(), 4),
    'cv_roc_auc_std': round(lr_cv.std(), 4)
}
models_info['LogisticRegression'] = {'type': 'Logistic Regression', 'coef_': lr.coef_[0].tolist()}

# 5.2 Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_cv = cross_val_score(rf, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc')
results['RandomForest'] = {
    'accuracy': round(accuracy_score(y_test, rf_pred), 4),
    'f1': round(f1_score(y_test, rf_pred), 4),
    'precision': round(precision_score(y_test, rf_pred), 4),
    'recall': round(recall_score(y_test, rf_pred), 4),
    'roc_auc': round(roc_auc_score(y_test, rf_proba), 4),
    'cv_roc_auc_mean': round(rf_cv.mean(), 4),
    'cv_roc_auc_std': round(rf_cv.std(), 4)
}
models_info['RandomForest'] = {
    'type': 'Random Forest',
    'feature_importance': rf.feature_importances_.tolist()
}

# 5.3 Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_proba = gb.predict_proba(X_test)[:, 1]
gb_cv = cross_val_score(gb, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc')
results['GradientBoosting'] = {
    'accuracy': round(accuracy_score(y_test, gb_pred), 4),
    'f1': round(f1_score(y_test, gb_pred), 4),
    'precision': round(precision_score(y_test, gb_pred), 4),
    'recall': round(recall_score(y_test, gb_pred), 4),
    'roc_auc': round(roc_auc_score(y_test, gb_proba), 4),
    'cv_roc_auc_mean': round(gb_cv.mean(), 4),
    'cv_roc_auc_std': round(gb_cv.std(), 4)
}
models_info['GradientBoosting'] = {
    'type': 'Gradient Boosting',
    'feature_importance': gb.feature_importances_.tolist()
}

# Select best model based on ROC-AUC
best_model_name = max(results, key=lambda k: results[k]['roc_auc'])
best_model_info = results[best_model_name]
print(f"\nBest model: {best_model_name} (ROC-AUC: {best_model_info['roc_auc']})")

# ── 6. Feature Importance from Best Model ────────────────────
perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, scoring='roc_auc')
feature_importance_list = []
for i, name in enumerate(encoded_feature_cols):
    feature_importance_list.append({
        'feature': name,
        'rf_importance': round(rf.feature_importances_[i], 4),
        'perm_importance': round(perm_imp.importances_mean[i], 4),
        'perm_std': round(perm_imp.importances_std[i], 4)
    })
feature_importance_list.sort(key=lambda x: x['perm_importance'], reverse=True)

# ── 7. Confusion Matrix for Best Model ───────────────────────
if best_model_name == 'RandomForest':
    best_pred = rf_pred
elif best_model_name == 'LogisticRegression':
    best_pred = lr_pred
else:
    best_pred = gb_pred

cm = confusion_matrix(y_test, best_pred)

# ── 8. Generate Sample Individual Predictions ────────────────
# Use entire dataset for individual predictions
X_all = X
X_all_scaled = scaler.transform(X_all)
rf_all_proba = rf.predict_proba(X_all)[:, 1]

# Risk level classification
def risk_level(prob):
    if prob < 0.15: return '매우 낮음'
    elif prob < 0.30: return '낮음'
    elif prob < 0.50: return '중간'
    elif prob < 0.70: return '높음'
    else: return '매우 높음'

sample_indices = np.random.choice(len(df_model), min(20, len(df_model)), replace=False)
sample_predictions = []
for idx in sample_indices:
    row = df_model.iloc[idx]
    prob = float(rf_all_proba[idx])
    sample_predictions.append({
        'age': int(row['AGE']),
        'sex': str(row['SEX']),
        'sbp': float(row['SBP']),
        'dbp': float(row['DBP']),
        'bmi': float(row['BMI']),
        'smoking': str(row['SMOKING']),
        'exercise': str(row['EXERCISE']),
        'actual_htn': str(row[TARGET_COL]),
        'predicted_prob': round(prob, 4),
        'risk_level': risk_level(prob)
    })

# ── 9. Population Statistics ─────────────────────────────────
pop_stats = {
    'total_records': len(df_model),
    'hypertension_count': int(y.sum()),
    'hypertension_pct': round(float(y.mean()) * 100, 1),
    'age_stats': {
        'mean': round(float(df_model['AGE'].mean()), 1),
        'min': int(df_model['AGE'].min()),
        'max': int(df_model['AGE'].max())
    },
    'sbp_stats': {
        'mean': round(float(df_model['SBP'].mean()), 1),
        'min': int(df_model['SBP'].min()),
        'max': int(df_model['SBP'].max()),
        'p25': round(float(df_model['SBP'].quantile(0.25)), 1),
        'p50': round(float(df_model['SBP'].quantile(0.50)), 1),
        'p75': round(float(df_model['SBP'].quantile(0.75)), 1)
    },
    'dbp_stats': {
        'mean': round(float(df_model['DBP'].mean()), 1),
        'min': int(df_model['DBP'].min()),
        'max': int(df_model['DBP'].max())
    },
    'bmi_stats': {
        'mean': round(float(df_model['BMI'].mean()), 1),
        'min': round(float(df_model['BMI'].min()), 1),
        'max': round(float(df_model['BMI'].max()), 1)
    }
}

# SBP vs Age by hypertension status
age_bins = [20, 30, 40, 50, 60, 70, 80, 100]
age_labels = ['20대', '30대', '40대', '50대', '60대', '70대', '80대이상']
df_model['AGE_GROUP'] = pd.cut(df_model['AGE'], bins=age_bins, labels=age_labels, right=False)

age_htn_stats = []
for grp in age_labels:
    subset = df_model[df_model['AGE_GROUP'] == grp]
    if len(subset) > 0:
        htn_sub = subset[subset['HTN_BINARY'] == 1]
        normal_sub = subset[subset['HTN_BINARY'] == 0]
        age_htn_stats.append({
            'age_group': grp,
            'total': len(subset),
            'htn_count': len(htn_sub),
            'htn_pct': round(len(htn_sub) / len(subset) * 100, 1),
            'avg_sbp_normal': round(float(normal_sub['SBP'].mean()), 1) if len(normal_sub) > 0 else 0,
            'avg_sbp_htn': round(float(htn_sub['SBP'].mean()), 1) if len(htn_sub) > 0 else 0
        })

# SEX-based stats
sex_htn_stats = []
for sex_val in df_model['SEX'].unique():
    subset = df_model[df_model['SEX'] == sex_val]
    htn_sub = subset[subset['HTN_BINARY'] == 1]
    sex_htn_stats.append({
        'sex': str(sex_val),
        'total': len(subset),
        'htn_count': len(htn_sub),
        'htn_pct': round(len(htn_sub) / len(subset) * 100, 1)
    })

# ── 10. Assemble Results ─────────────────────────────────────
output = {
    'models': results,
    'best_model': best_model_name,
    'best_model_metrics': best_model_info,
    'feature_importance': feature_importance_list,
    'feature_names': encoded_feature_cols,
    'confusion_matrix': cm.tolist(),
    'sample_predictions': sample_predictions,
    'population_stats': pop_stats,
    'age_htn_stats': age_htn_stats,
    'sex_htn_stats': sex_htn_stats,
    'encoders': encoders
}

OUTPUT_PATH = r'C:\Users\USER\Desktop\PHR분석실무\bp_prediction\results.json'
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ Results saved to: {OUTPUT_PATH}")
print(f"Best model: {best_model_name}")
for metric, val in best_model_info.items():
    print(f"  {metric}: {val}")
