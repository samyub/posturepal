import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

from features import extract_features
from name import build_dataset


print("📦 Loading dataset...")
data = build_dataset()

X = []
y = []

for sample in data:
    feats = extract_features(sample["landmarks"])
    X.append(feats)
    y.append(sample["label"])

X = np.array(X)
y = np.array(y)

print("Feature shape:", X.shape)

# --- Encode labels ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- Train / test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# --- Train classifier ---
print("🧠 Training classifier...")
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# --- Evaluate ---
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {acc * 100:.2f}%")
print("\n📊 Classification report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# --- Save model ---
joblib.dump(clf, "pose_classifier.joblib")
joblib.dump(label_encoder, "label_encoder.joblib")

print("\n💾 Model saved:")
print(" - pose_classifier.joblib")
print(" - label_encoder.joblib")
