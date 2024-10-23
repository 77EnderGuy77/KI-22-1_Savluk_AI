import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from sklearn.metrics import silhouette_score

# 1. Завантаження даних з CSV
data = pd.read_csv('./lab01.csv')
x = data['x'].values
y = data['y'].values
points = np.column_stack((x, y))

# 2. Метод зсуву середнього для визначення кількості кластерів
bandwidth = estimate_bandwidth(points, quantile=0.2)
mean_shift = MeanShift(bandwidth=bandwidth)
mean_shift.fit(points)
cluster_centers = mean_shift.cluster_centers_
n_clusters_ = len(cluster_centers)

print(f"Кількість кластерів (метод зсуву середнього): {n_clusters_}")

# 3. Оцінка score для різних кількостей кластерів (від 2 до 15)
scores = []
cluster_range = range(2, 16)
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(points)
    labels = kmeans.labels_
    score = silhouette_score(points, labels)
    scores.append(score)

# 4. Використання оптимальної кількості кластерів для k-середніх
optimal_clusters = cluster_range[np.argmax(scores)]
kmeans = KMeans(n_clusters=optimal_clusters)
kmeans.fit(points)
kmeans_labels = kmeans.labels_

# --- Графічне виведення ---

# Створюємо загальну фігуру з 4 підграфіками (2x2)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 1. Вихідні точки на площині (верхній лівий)
axs[0, 0].scatter(x, y)
axs[0, 0].set_title('Вихідні точки')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('y')

# 2. Центри кластерів (метод зсуву середнього) (верхній правий)
axs[0, 1].scatter(x, y, c='blue', label='Вихідні точки')
axs[0, 1].scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Центри кластерів')
axs[0, 1].set_title('Центри кластерів (метод зсуву середнього)')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('y')
axs[0, 1].legend()

# 3. Бар-діаграма score(number of clusters) (нижній правий)
axs[1, 1].bar(cluster_range, scores)
axs[1, 1].set_title('Оцінка silhouette для різної кількості кластерів')
axs[1, 1].set_xlabel('Кількість кластерів')
axs[1, 1].set_ylabel('Silhouette Score')

# 4. Кластеризовані дані з областями кластеризації (нижній лівий)
scatter = axs[1, 0].scatter(x, y, c=kmeans_labels, cmap='viridis')
axs[1, 0].set_title(f'Кластеризовані дані (K-середніх, {optimal_clusters} кластерів)')
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('y')

# Відображаємо всі графіки одночасно
plt.tight_layout()
plt.show()
