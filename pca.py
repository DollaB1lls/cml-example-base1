import os
import numpy as np
from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_images_from_folders(folder, image_size=(64, 64), limit_per_class=150):
    images = []
    labels = []
    
    for subfolder_name in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            print(f"Loading images from {subfolder_name}")
            count = 0
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpg') and count < limit_per_class:
                    img_path = os.path.join(subfolder_path, filename)
                    img = Image.open(img_path)
                    img = img.convert('L')
                    img = img.resize(image_size)
                    img_array = np.array(img).flatten()
                    images.append(img_array)
                    labels.append(subfolder_name)
                    count += 1
            print(f"Loaded {count} images from {subfolder_name}")
    
    print(f"Total images loaded: {len(images)}")
    return np.array(images), np.array(labels)

folder_path = 'C:\VS Code\CML\planetsandmoons'
X, y = load_images_from_folders(folder_path)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=11)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 2], X_pca[:, 1], edgecolor='k', s=50) #change variables here
plt.title('PCA of Planet/Moon Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()