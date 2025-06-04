import cv2
import numpy as np
from PIL import Image


class AvgRGB:
    def __init__(self):
        self.count = 0
        self.sum = [0, 0, 0]
        self.avg = [0, 0, 0]
        self.max = [float('-inf')] * 3
        self.min = [float('inf')] * 3
        self.current = None

    def update(self, current):
        if isinstance(current, int):  # Если вдруг приходит ч/б пиксель
            current = (current, current, current)

        self.current = current
        self.count += 1

        for i in range(3):  # Обновляем по каждому каналу R, G, B
            self.sum[i] += current[i]
            self.avg[i] = self.sum[i] / self.count
            self.max[i] = max(self.max[i], current[i])
            self.min[i] = min(self.min[i], current[i])

    def get_avg(self):
        return tuple(self.avg) if self.count > 0 else (0, 0, 0)



import cv2
import numpy as np
from PIL import Image

def find_contours_1px(image_path):
    """
    Возвращает PIL.Image с ровными, замкнутыми, 1px-внешними контурами всех клеток
    на исходном кровяном снимке (включая те, что заходят за границу).
    Алгоритм:
      1. Добавляем 2px-рамку (паддинг) по периметру.
      2. Переводим в LAB и берём канал A для цветовой сегментации.
      3. Бинаризуем A (Otsu) → «грубая» маска.
      4. Morphology (Opening 3×3, Closing 3×3, Closing 5×5×2) → замкнутая маска.
      5. (Опционально) OR с Canny (пороги по Otsu) на размытом CLAHE-грее.
      6. Обрезка мелких объектов по площади (адаптивная).
      7. Заливка оставшихся контуров.
      8. Эрозия 1px внутрь + вычитание → ровный 1px край.
      9. Рисуем этот край белым в чёрном фоне, снимаем паддинг.
    """

    # 1) Загрузка и паддинг (2px)
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
    padded = cv2.copyMakeBorder(orig, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 2) Переводим в LAB и берём канал A (отлично отделяет пурпурные клетки)
    lab = cv2.cvtColor(padded, cv2.COLOR_BGR2LAB)
    _, A_channel, _ = cv2.split(lab)

    # 3) Otsu-бинаризация канала A: retval (порог), mask (бинарная маска)
    otsu_thresh, mask_A = cv2.threshold(
        A_channel,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # → mask_A == 255 там, где канал A выше порога, то есть там, где пурпурные/фиолетовые области (клетки).

    # 4) Morphological «очистка» маски (маска из A_channel)
    kernel3 = np.ones((3, 3), np.uint8)
    mask_open = cv2.morphologyEx(mask_A, cv2.MORPH_OPEN, kernel3, iterations=1)
    mask_closed = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel3, iterations=1)

    # Ещё один проход Closing (5×5, iterations=2) для гарантированного «замыкания» пробоин
    kernel5 = np.ones((5, 5), np.uint8)
    mask_closed2 = cv2.morphologyEx(mask_closed, cv2.MORPH_CLOSE, kernel5, iterations=2)

    # (Опционально) шаг 5: объединяем с границами Canny (пороги по Otsu на размытом CLAHE)
    # Если вам нужно «усилить» тонкие контуры, можно раскомментировать следующий блок:
    """
    #   a) CLAHE + GaussianBlur (5×5, 0.5)
    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray_clahe, (5,5), 0.5)
    #   b) Otsu-пороги для Canny
    otsu2, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lower = int(max(0, 0.5 * otsu2))
    upper = int(min(255, 1.0 * otsu2))
    edges = cv2.Canny(blurred, lower, upper)
    #   c) Объединение с mask_closed2
    mask_combined = cv2.bitwise_or(mask_closed2, edges)
    #   d) Ещё раз Closing (5×5, iterations=2)
    mask_final = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel5, iterations=2)
    """
    # Если не включаем Canny, просто работаем с mask_closed2:
    mask_final = mask_closed2.copy()

    # 6) Фильтрация по адаптивному порогу площади:
    #    img_area = h×w, порог = max(100px, 0.01% от всей площади)
    h, w = mask_final.shape[:2]
    img_area = h * w
    adaptive_min_area = max(100, img_area * 0.0001)

    contours, _ = cv2.findContours(
        mask_final,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) >= adaptive_min_area]

    # 7) Заливаем отфильтрованные контуры → получаем «сплошную» маску_filled
    mask_filled = np.zeros_like(mask_final)
    cv2.drawContours(mask_filled, filtered, -1, 255, thickness=cv2.FILLED)

    # 8) Эрозия «внутрь» на 1px
    eroded = cv2.erode(mask_filled, kernel3, iterations=1)

    # 9) Вычитаем эродированную маску из оригинальной «сплошной» → ровная «корона» толщиной 1px
    contour_mask = cv2.subtract(mask_filled, eroded)

    # 10) Рисуем этот контур (где contour_mask == 255) белым (255,255,255) на «чёрном» фоне
    result = np.zeros_like(padded)
    result[contour_mask == 255] = (255, 255, 255)

    # 11) Убираем искусственный паддинг 2px и возвращаем оригинальный размер
    h0, w0 = orig.shape[:2]
    result_cropped = result[2:2 + h0, 2:2 + w0]

    # 12) Конвертируем в PIL.Image и возвращаем
    result_rgb = cv2.cvtColor(result_cropped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


import cv2
import numpy as np
from PIL import Image

def find_contours_1px_universal(image_path):
    """
    Обновлённая универсальная функция для поиска ровных 1px внешних контуров клеток
    (любого цвета) на белом/светлом фоне.

    Алгоритм:
      1) Добавляем 2px паддинг вокруг изображения, чтобы контуры на границе замыкались.
      2) Переводим в grayscale и усиливаем локальный контраст (CLAHE).
      3) Слегка сглаживаем (medianBlur + GaussianBlur) для удаления мелкого шума.
      4) Делаем две бинаризации:
         - adaptiveThreshold (локальная пороговая бинаризация, inverted),
         - OtsuThresh (глобальная пороговая бинаризация, inverted).
      5) Объединяем (OR) обе карты, получая «грубую» бинарную маску всех объектов.
      6) Морфология (opening 5×5 → closing 3×3 → closing 5×5×2) для «очистки» маски
         и гарантированного замыкания всех контуров.
      7) findContours **на маске mask_closed** (не инвертируем!), т.к. объекты там уже белые.
      8) Фильтрация контуров только по площади (adaptive_min_area).
      9) Заливаем (drawContours FILLED) все отфильтрованные контуры → получаем
         «сплошную» замкнутую маску_filled.
     10) Эрозия (kernel 3×3, iterations=1) «сдвигает» границу внутрь на 1px.
     11) (mask_filled − eroded) даёт ровный 1px-контур («маска-корона»).
     12) Рисуем эти пиксели (где маска-корона=255) белым цветом в чёрном фоне,
         обрезаем паддинг 2px и получаем итоговый объект.
    """

    # 1) Загрузка и паддинг (2 px)
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
    padded = cv2.copyMakeBorder(orig, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 2) Переводим в grayscale + CLAHE
    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # 3) Убираем мелкий шум: сначала median, затем Gaussian
    median = cv2.medianBlur(gray_clahe, 5)
    blurred = cv2.GaussianBlur(median, (5, 5), 0.5)

    # 4a) Adaptive Threshold (локально инвертируем), блок 15×15, C=4
    mask_adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        4
    )
    # 4b) Otsu Threshold (глобально инвертируем)
    _, mask_otsu = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 5) Объединяем adaptive + Otsu (логическое OR)
    mask_combined = cv2.bitwise_or(mask_adaptive, mask_otsu)

    # 6) Morphology: «очищаем» и замыкаем контура
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)

    # 6a) Opening 5×5 (удаляет мелкий шум)
    mask_open = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel5, iterations=1)
    # 6b) Closing 3×3 (заливает небольшие дыры внутри объектов)
    mask_close1 = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel3, iterations=1)
    # 6c) Closing 5×5 дважды (гарантирует замыкание всех разрывов)
    mask_closed = cv2.morphologyEx(mask_close1, cv2.MORPH_CLOSE, kernel5, iterations=2)

    # 7) Ищем все внешние контуры на маске mask_closed (тут объекты уже белые)
    contours, _ = cv2.findContours(
        mask_close1,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 8) Фильтрация по адаптивному порогу площади:
    h, w = mask_closed.shape[:2]
    img_area = h * w
    adaptive_min_area = max(100, img_area * 0.0005)  # минимум 100 px² или 0.05% от кадра
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) >= adaptive_min_area]

    # 9) Заливаем (FILLED) все отфильтрованные контуры в mask_filled
    mask_filled = np.zeros_like(mask_closed)
    cv2.drawContours(mask_filled, filtered, -1, 255, thickness=cv2.FILLED)

    # 10) Эрозируем mask_filled «внутрь» на 1px
    eroded = cv2.erode(mask_filled, kernel3, iterations=1)

    # 11) Вычитаем eroded из mask_filled → ровный 1px-контур
    contour_mask = cv2.subtract(mask_filled, eroded)

    # 12) Рисуем contour_mask (где ==255) белым (255,255,255) на чёрном фоне
    result = np.zeros_like(padded)
    result[contour_mask == 255] = (255, 255, 255)

    # 13) Обрезаем паддинг 2px, возвращаем оригинальный размер
    h0, w0 = orig.shape[:2]
    result_cropped = result[2:2 + h0, 2:2 + w0]

    # 14) Конвертируем BGR→RGB и возвращаем PIL.Image
    result_rgb = cv2.cvtColor(result_cropped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


def find_contours_1px_universal_refined(image_path):
    """
    Универсальная функция для поиска и отрисовки ровных 1px внешних контуров клеток крови
    (любого цвета) с минимизацией ложных шумовых контуров.
    """

    # 1) Загружаем исходник и добавляем 2px паддинг (чтобы замкнуть контуры на краю)
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
    padded = cv2.copyMakeBorder(orig, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 2) Переводим в серый и применяем CLAHE (локальное выравнивание контраста)
    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # 3) Медиа́нное размытие для удаления мелких «пятнышек»-шумов
    median = cv2.medianBlur(gray_clahe, 5)

    # 4) Небольшое GaussianBlur (для более плавной Otsu-бинаризации)
    blurred = cv2.GaussianBlur(median, (5, 5), 0.5)

    # 5a) Adaptive Threshold (GAUSSIAN_C) с увеличенным блоком (15×15) и C = 4
    mask_adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        4
    )

    # 5b) Otsu Threshold (глобальная бинаризация, inverted)
    _, mask_otsu = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 6) Объединяем adaptive + Otsu (bitwise OR)
    mask_combined = cv2.bitwise_or(mask_adaptive, mask_otsu)

    # 7) Morphology «очистка»:
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)

    #   a) Opening 5×5 (удаляем мелкие пятнышки) — iterations=1
    mask_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel5, iterations=1)

    #   b) Closing 3×3 (заливка небольших дыр)
    mask_close1 = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel3, iterations=1)

    #   c) Closing 5×5 (дополнительное замыкание) — iterations=2
    mask_closed = cv2.morphologyEx(mask_close1, cv2.MORPH_CLOSE, kernel5, iterations=2)

    # 8) Инвертируем маску: чтобы клетки (объекты) были белыми (255), фон — чёрным (0)
    mask_inv = cv2.bitwise_not(mask_closed)

    # 9) Ищем все внешние контуры (RETR_EXTERNAL)
    contours, _ = cv2.findContours(
        mask_inv,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 10) Фильтрация по площади и по «extent» (отношению area к bounding box area)
    h, w = mask_inv.shape[:2]
    img_area = h * w
    # adaptive_min_area = минимум 100 px² или 0.01% от площади
    adaptive_min_area = max(100, img_area * 0.0001)
    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < adaptive_min_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        rect_area = cw * ch
        if rect_area <= 0:
            continue
        extent = float(area) / rect_area
        # У клеток extent обычно около 0.6–0.9. Ушума может быть много или очень маленьким (extent < 0.3).
        if extent < 0.3 or extent > 0.95:
            continue
        filtered.append(cnt)

    # 11) Заливаем (FILLED) все отфильтрованные контуры → получаем «сплошную» маску_filled
    mask_filled = np.zeros_like(mask_inv)
    cv2.drawContours(mask_filled, filtered, -1, 255, thickness=cv2.FILLED)

    # 12) Эрозия «внутрь» на 1px (ядро 3×3, iterations=1)
    eroded = cv2.erode(mask_filled, kernel3, iterations=1)

    # 13) «Сплошная» − «эродированная» → ровная «корона» толщиной ровно 1px
    contour_mask = cv2.subtract(mask_filled, eroded)

    # 14) Рисуем contour_mask (где == 255) белым (255,255,255) на чёрном фоне (3 канала)
    result = np.zeros_like(padded)
    result[contour_mask == 255] = (255, 255, 255)

    # 15) Обрезаем добавленные 2 px паддинга, возвращаем исходный размер
    h0, w0 = orig.shape[:2]
    result_cropped = result[2:2 + h0, 2:2 + w0]

    # 16) Конвертируем BGR→RGB и возвращаем как PIL.Image
    result_rgb = cv2.cvtColor(result_cropped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


# result = find_contours_1px('img/moon.jpg')
#
# cv2.imshow('Contours 1px', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
