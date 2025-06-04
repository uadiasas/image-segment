import cv2 as cv
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt

import test2

# Установка бэкенда для корректного отображения в PyCharm
import matplotlib
matplotlib.use('TkAgg')

import sys
sys.setrecursionlimit(10000)

def is_similar_rgb(pixel1, pixel2, treshold,  cx, cy, nx, ny): # сделать так, чтобы цикл шёл к самому яркому соседу
    r1, g1, b1 = pixel1
    r2, g2, b2 = pixel2
    distance = ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5
    # print(f"Дистанция между пикселями x1={cx},y1={cy}  x2={nx},y2={ny}     {pixel1} и  {pixel2}", distance)
    return distance <= treshold

def marker_of_one_object2(diff_img, x, y, current_max_x, current_max_y, visited, treshold):
    stack = [(x, y)]
    max_x, max_y = current_max_x, current_max_y
    min_x, min_y = current_max_x, current_max_y
    i = 0
    while stack:
        i += 1
        cx, cy = stack.pop()
        if (cx, cy) in visited:
            continue
        visited.add((cx, cy))
        max_x = max(max_x, cx)
        max_y = max(max_y, cy)
        min_x = min(min_x, cx)
        min_y = min(min_y, cy)
        for nx, ny in [(cx + 1, cy), (cx + 1, cy + 1), (cx, cy + 1), (cx - 1, cy + 1), (cx - 1, cy),
                       (cx - 1, cy - 1), (cx, cy - 1), (cx + 1, cy - 1), ]:
            if 0 <= nx < diff_img.width and 0 <= ny < diff_img.height:
                pixel_value = diff_img.getpixel((nx, ny))
                r, g, b = pixel_value
                initial_pixel = diff_img.getpixel((cx, cy))
                if is_similar_rgb(pixel_value, initial_pixel, treshold, cx, cy, nx, ny):
                    stack.append((nx, ny))
    return max_x, max_y, min_x, min_y, i

# Открываем изображение
image_path = "img/blood4.png"
image = Image.open(image_path)
image = image.filter(ImageFilter.MedianFilter(size=7))
image_for_markers = cv.imread(image_path)

# Определяем маску изображения
# a = 0.01
# image_mask = np.array([[a / 4, (1 - a)/4, a/4],
#                        [(1 - a)/4, -1, (1 - a)/4],
#                        [a / 4, (1 - a)/4, a/4]])
# image_mask = 4 / (a + 1) * image_mask

# Получаем размеры изображения
width, height = image.size
new_image = Image.new("RGB", (width, height))
#
# # Преобразуем изображение в список RGB и применяем маску
# image_list = []
# for y in range(1, height - 1):
#     row = []
#     for x in range(1, width - 1):
#         sr = sg = sb = 0
#         for i in range(3):
#             for j in range(3):
#                 pixel_value = image.getpixel((x + i - 1, y + j - 1))
#                 r, g, b = pixel_value
#                 sr += image_mask[i, j] * r
#                 sg += image_mask[i, j] * g
#                 sb += image_mask[i, j] * b
#         sr = max(0, min(255, int(sr)))
#         sg = max(0, min(255, int(sg)))
#         sb = max(0, min(255, int(sb)))
#         new_image.putpixel((x, y), (sr, sg, sb))
#         row.append((sr, sg, sb))
#     image_list.append(row)
#
# new_image.save("output_rgb_image_diff.png")
# new_image.show()

# Загружаем обработанное изображение для выделения маркеров

diff_img = Image.open("output_rgb_image_diff.png")
# diff_img = diff_img.filter(ImageFilter.GaussianBlur(3))
# image2 = cv.imread(image_path)
diff_img = test2.find_contours_1px_universal(image_path)

# # 💉 Морфология — заливаем дыры в границах
# diff_cv = np.array(diff_img)
# diff_cv = cv.cvtColor(diff_cv, cv.COLOR_RGB2GRAY)
#
# # Делаем бинарную маску (всё, что выше 50 — белое, остальное — чёрное)
# _, binary = cv.threshold(diff_cv, 50, 255, cv.THRESH_BINARY)
#
# # Закрываем дырки в границах
# kernel = np.ones((7, 7), np.uint8)
# closed = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
#
# # Обратно в формат, который понимает твоя функция (PIL RGB)
# diff_img = Image.fromarray(cv.cvtColor(closed, cv.COLOR_GRAY2RGB))

diff_img.show()
width_diff_img, height_diff_img = diff_img.size


# Создаем маркерную матрицу и список для координат
markers = np.zeros((height_diff_img, width_diff_img), dtype=np.int32)
rect_about_object = np.zeros((height_diff_img, width_diff_img), dtype=np.int32)
object_markers = []  # Список для хранения координат маркеров объектов
background_markers = []

# Определение маркеров (объектов и фона)
threshold_in = 200
threshold_out = 50
contour = False
object_index = 2
visited = set()

for y in range(2, height_diff_img - 2):
    for x in range(2, width_diff_img - 2):
        pixel_value = diff_img.getpixel((x, y))
        r, g, b = pixel_value
        if r + g + b > threshold_in and not contour and (x, y) not in visited:

            max_x, max_y, min_x, min_y, count_pix = marker_of_one_object2(diff_img, x, y, x, y, visited, treshold=threshold_out)
            if count_pix >= 3:
                # markers[y - 2, x - 2] = 1
                # background_markers.append((x - 2, y - 2))
                marker_x = (max_x + min_x) // 2
                marker_y = (max_y + min_y) // 2
                markers[marker_y, marker_x] = object_index  # Устанавливаем маркер объекта
                object_markers.append((object_index, marker_x, marker_y))  # Сохраняем координаты
                object_index += 1
                print(f"Заполняем от min_y={min_y} до max_y={max_y}, от min_x={min_x} до max_x={max_x}")
                rect_about_object[min_y-1:max_y + 1, min_x-1:max_x + 1] = 1
                contour = True
        elif r + g + b == 0 and contour:
            # markers[y + 2, x + 2] = 1  # Фон
            # background_markers.append((x + 2, y + 2))
            contour = False

# 1. Находим контуры прямоугольников объектов
contours, _ = cv.findContours(rect_about_object.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 2. Собираем центры всех прямоугольников
centers = []
for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    cx = x + w // 2
    cy = y + h // 2
    centers.append((cx, cy))

# 3. Для каждой уникальной пары центров — ставим маркер между ними
for i in range(len(centers)):
    for j in range(i + 1, len(centers)):
        cx1, cy1 = centers[i]
        cx2, cy2 = centers[j]

        mx = (cx1 + cx2) // 2
        my = (cy1 + cy2) // 2

        # Только если точка лежит ВНЕ объектов (в нуле)
        if rect_about_object[my, mx] == 0 and markers[my, mx] == 0:
            markers[my, mx] = 1
            background_markers.append((mx, my))



# Выводим координаты маркеров объектов
print("\nКоординаты маркеров объектов:")
for idx, mx, my in object_markers:
    print(f"Маркер объекта {idx} установлен в координатах: x={mx}, y={my}")
print("\nКоординаты маркеров фона:")
for mx, my in background_markers:
    print(f"Маркер фона установлен в координатах: x={mx}, y={my}")

# Создаём отдельное изображение с маркерами
image_with_markers_only = cv.imread(image_path).copy()
for idx, mx, my in object_markers:
    cv.circle(image_with_markers_only, (mx, my), 5, (0, 255, 0), -1)  # Зелёные маркеры объектов
for mx, my in background_markers:
    cv.circle(image_with_markers_only, (mx, my), 5, (255, 0, 0), -1)  # Красные маркеры фона
plt.imshow(cv.cvtColor(image_with_markers_only, cv.COLOR_BGR2RGB))
plt.title('Image with Object and Background Markers')
plt.axis('off')
plt.show()

# Применяем алгоритм Watershed
image_for_watershed = cv.imread(image_path)
markers = cv.watershed(image_for_watershed, markers)

# Визуализация результатов
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
result = np.zeros_like(image_for_watershed, dtype=np.uint8)
for marker1 in np.unique(markers):
    if marker1 == -1:
        continue
    mask = markers == marker1
    color = np.random.randint(0, 255, size=3, dtype=np.uint8)
    result[mask] = color

axes[1].imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
axes[1].set_title('Результат сегментации методом водоразделов')
axes[1].axis('off')
plt.show()

image_for_watershed[markers == -1] = [255, 0, 0]  # Границы красным
plt.imshow(cv.cvtColor(image_for_watershed, cv.COLOR_BGR2RGB))
plt.title('Segmented Image with Watershed')
plt.show()

# Наносим точки маркеров на изображение
for y in range(1, height_diff_img - 1):
    for x in range(1, width_diff_img - 1):
        if markers[y, x] == 2:
            cv.circle(image_for_markers, (x, y), 2, (0, 150, 0), -1)
        elif markers[y, x] == 3:
            cv.circle(image_for_markers, (x, y), 2, (0, 200, 0), -1)
        elif markers[y, x] == 4:
            cv.circle(image_for_markers, (x, y), 2, (0, 255, 0), -1)
        elif markers[y, x] == 1:
            cv.circle(image_for_markers, (x, y), 2, (255, 0, 0), -1)

plt.imshow(cv.cvtColor(image_for_markers, cv.COLOR_BGR2RGB))
plt.title('Image with Markers')
plt.show()
print(object_index - 1)