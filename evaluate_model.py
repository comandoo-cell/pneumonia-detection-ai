from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# 1️⃣ إعداد المسارات
model_path = 'best_model_NEW_TEST.h5'   # نموذج آخر للتجربة
test_dir = 'chest_xray/test'   # المسار الصحيح لمجلد الاختبار

# 2️⃣ تحميل النموذج
model = load_model(model_path)
print("✅ النموذج تم تحميله بنجاح!")

# 3️⃣ تجهيز بيانات الاختبار
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),  # غيّرها إذا استخدمت حجم آخر أثناء التدريب
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# 4️⃣ إجراء التنبؤ
Y_pred = model.predict(test_generator)
y_pred = (Y_pred > 0.5).astype(int).flatten()
y_true = test_generator.classes

# 5️⃣ طباعة النتائج الأساسية
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))

# 6️⃣ مصفوفة الالتباس
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Pneumonia'],
            yticklabels=['Normal', 'Pneumonia'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 7️⃣ منحنى ROC و AUC
fpr, tpr, thresholds = roc_curve(y_true, Y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
