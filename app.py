import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import json
from pathlib import Path

# ============================================================================
# КОНФИГУРАЦИЯ МОДЕЛИ И ПАРАМЕТРОВ
# ============================================================================

# Путь к файлу весов модели
MODEL_PATH = "best.pt"

# Классы дорожных дефектов в точном порядке (индекс соответствует классу)
CLASS_NAMES = {
    0: "longitudinal_crack",
    1: "transverse_crack",
    2: "alligator_crack",
    3: "other_corruption",
    4: "pothole",
}

# Цвета для каждого класса (BGR формат для OpenCV)
CLASS_COLORS = {
    0: (255, 0, 0),        # Alligator - синий
    1: (0, 255, 0),        # Lateral-Crack - зелёный
    2: (0, 0, 255),        # Longitudinal-Crack - красный
    3: (255, 255, 0),      # pothole - голубой
    4: (255, 0, 255)       # other_damage - пурпурный
}

# Параметры детекции
CONFIDENCE_THRESHOLD = 0.15
IMG_SIZE = 640

# Глобальная переменная для хранения модели (загружается один раз при запуске)
model = None

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ МОДЕЛИ
# ============================================================================

def load_model():
    """
    Загружает модель YOLOv8 при запуске приложения.
    Модель загружается один раз в памяти и переиспользуется для всех предсказаний.
    """
    global model
    try:
        model = YOLO(MODEL_PATH)
        print(f"✓ Модель успешно загружена: {MODEL_PATH}")
        print(f"  Классы: {list(CLASS_NAMES.values())}")
        print(f"  Порог уверенности: {CONFIDENCE_THRESHOLD}")
        return True
    except Exception as e:
        print(f"✗ Ошибка при загрузке модели: {e}")
        return False

# ============================================================================
# ФУНКЦИЯ ДЕТЕКЦИИ И ОБРАБОТКИ РЕЗУЛЬТАТОВ
# ============================================================================

def detect_road_defects(input_image, conf=CONFIDENCE_THRESHOLD):
    """
    Основная функция для детекции дорожных дефектов.
    
    Args:
        input_image: изображение в формате numpy array (загруженное через Gradio)
    
    Returns:
        кортеж из трёх элементов:
        - annotated_image: изображение с нарисованными боксами и метками
        - summary_text: текстовое резюме с таблицей результатов
        - json_output: JSON с координатами и уверенностью детекций
    """
    
    if model is None:
        return None, "❌ Ошибка: Модель не загружена", "{}"
    
    try:
        # Проверка входного изображения
        if input_image is None:
            return None, "❌ Ошибка: Изображение не загружено", "{}"
        
        # Преобразование изображения в RGB если необходимо
        # Gradio возвращает изображение в формате RGB по умолчанию
        if len(input_image.shape) == 2:
            # Если grayscale, конвертируем в RGB
            image_rgb = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
        elif len(input_image.shape) == 3 and input_image.shape[2] == 4:
            # Если RGBA, конвертируем в RGB
            image_rgb = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)
        elif len(input_image.shape) == 3 and input_image.shape[2] == 3:
            # Если уже RGB (от Gradio), используем как есть
            image_rgb = input_image
        else:
            # На всякий случай, если неизвестный формат
            image_rgb = input_image
        
        # Запуск модели на изображении
        results = model(image_rgb, conf=conf, iou=0.5, imgsz=IMG_SIZE, verbose=False)
        
        # Инициализация результатов
        detections_list = []
        class_counts = {i: 0 for i in range(len(CLASS_NAMES))}
        total_detections = 0
        
        # Подготовка изображения для отрисовки
        annotated_image = image_rgb.copy()
        
        # Обработка результатов детекции
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            # Итерация по каждому обнаруженному объекту
            for i, box in enumerate(boxes):
                # Извлечение координат бокса (нормализованные)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Преобразование в целые координаты пикселей
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Валидация координат - они должны быть в пределах размеров изображения
                h, w = image_rgb.shape[:2]
                if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
                    # Пропускаем некорректный бокс
                    continue
                
                # Получение класса и уверенности
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_name = CLASS_NAMES.get(class_id, "Unknown")
                color = CLASS_COLORS.get(class_id, (255, 255, 255))
                
                # Добавление информации в список детекций
                detections_list.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                })
                
                # Обновление счётчика класса
                class_counts[class_id] += 1
                total_detections += 1
                
                # ===== ОТРИСОВКА БОКСОВ И МЕТОК =====
                # Рисование прямоугольника боксa
                thickness = 2
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
                
                # Подготовка текста метки
                label_text = f"{class_name} {confidence:.1%}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                
                # Получение размера текста
                text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
                text_width, text_height = text_size
                
                # Координаты фонового прямоугольника под текст
                bg_x1 = x1
                bg_y1 = y1 - text_height - 6
                bg_x2 = x1 + text_width + 4
                bg_y2 = y1
                
                # Обработка выхода за границы изображения
                if bg_y1 < 0:
                    bg_y1 = y2
                    bg_y2 = y2 + text_height + 6
                
                # Рисование заполненного прямоугольника (фон для текста)
                cv2.rectangle(annotated_image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
                
                # Рисование текста метки
                text_color = (255, 255, 255)  # Белый текст
                cv2.putText(annotated_image, label_text, (x1 + 2, y1 - 4), 
                           font, font_scale, text_color, font_thickness)
        
        # ===== ФОРМИРОВАНИЕ ТЕКСТОВОГО РЕЗЮМЕ =====
        if total_detections == 0:
            summary_text = "✓ Дефекты не обнаружены"
        else:
            # Создание таблицы результатов
            summary_lines = [
                f"📊 РЕЗУЛЬТАТЫ ДЕТЕКЦИИ",
                f"{'='*40}",
                f"Всего обнаружено дефектов: {total_detections}",
                f"{'='*40}",
                f"{'Класс':<20} {'Кол-во':<10}",
                f"{'-'*30}"
            ]
            
            # Добавление строк таблицы для каждого класса
            for class_id in sorted(class_counts.keys()):
                count = class_counts[class_id]
                class_name = CLASS_NAMES[class_id]
                if count > 0:
                    summary_lines.append(f"{class_name:<20} {count:<10}")
            
            summary_text = "\n".join(summary_lines)
        
        # ===== ФОРМИРОВАНИЕ JSON ВЫВОДА =====
        json_output = {
            "total_detections": total_detections,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "detections": detections_list
        }
        json_str = json.dumps(json_output, ensure_ascii=False, indent=2)
        
        # Конвертирование результирующего изображения из RGB в BGR для отображения
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        
        return annotated_image_bgr, summary_text, json_str
        
    except Exception as e:
        error_msg = f"❌ Ошибка при обработке изображения: {str(e)}"
        print(error_msg)
        return input_image, error_msg, "{}"

# ============================================================================
# СОЗДАНИЕ GRADIO ИНТЕРФЕЙСА
# ============================================================================

def create_interface():
    """
    Создаёт веб-интерфейс Gradio для детекции дорожных дефектов.
    """
    
    with gr.Blocks(title="Детекция дорожных дефектов") as interface:
        
        # ===== ЗАГОЛОВОК И ОПИСАНИЕ =====
        gr.Markdown("""
        # 🛣️ Система детекции дорожных дефектов
        
        ### Описание системы:
        Данная система использует нейронную сеть YOLOv8 для автоматического обнаружения 
        и классификации дефектов дорожного покрытия. Система обучена на датасете `rdd2022` и
        включает следующие классы (порядок важен и берётся из `data.yaml`):
        
        - **longitudinal_crack** — продольная трещина
        - **transverse_crack** — поперечная трещина
        - **alligator_crack** — сетка (аллигаторовая) трещин
        - **other_corruption** — прочие повреждения покрытия
        - **pothole** — выбоина
        
        ### Как использовать:
        1. Загрузите изображение дорожного покрытия в форму ниже
        2. Система автоматически обнаружит дефекты
        3. Результаты будут показаны в виде аннотированного изображения и таблицы
        """)
        
        # ===== РАЗДЕЛ ЗАГРУЗКИ ИЗОБРАЖЕНИЯ =====
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📤 Входные данные")
                input_image = gr.Image(
                    label="Загрузить изображение",
                    type="numpy",
                    sources=["upload", "webcam"],
                    interactive=True
                )
                conf_slider = gr.Slider(
                    minimum=0.05,
                    maximum=0.5,
                    value=0.15,
                    step=0.05,
                    label="Порог уверенности"
                )
                run_button = gr.Button(
                    "🔍 Запустить детекцию",
                    variant="primary",
                    scale=1
                )
        
        # ===== РАЗДЕЛ РЕЗУЛЬТАТОВ =====
        gr.Markdown("## 📊 Результаты")
        
        with gr.Row():
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Аннотированное изображение с боксами",
                    type="numpy"
                )
            
            with gr.Column(scale=1):
                summary_output = gr.Textbox(
                    label="Резюме результатов",
                    lines=10,
                    interactive=False,
                    max_lines=15
                )
        
        # ===== РАЗДЕЛ JSON ВЫВОДА =====
        with gr.Row():
            json_output = gr.Textbox(
                label="JSON вывод (детали детекций)",
                lines=10,
                interactive=False,
                max_lines=20
            )
        
        # ===== ПРИВЯЗКА КНОПКИ К ФУНКЦИИ =====
        run_button.click(
            fn=detect_road_defects,
            inputs=[input_image, conf_slider],
            outputs=[output_image, summary_output, json_output]
        )
        
        # ===== ИНФОРМАЦИЯ О КЛАССАХ =====
        gr.Markdown("""
        ### 🎨 Схема цветов классов:
        | Класс | Цвет |
        |-------|------|
        | longitudinal_crack | 🔴 Красный |
        | transverse_crack | 🔵 Синий |
        | alligator_crack | 🟢 Зелёный |
        | other_corruption | 🟣 Пурпурный |
        | pothole | 🟨 Жёлтый |
        """)
    
    return interface

# ============================================================================
# ТОЧКА ВХОДА В ПРИЛОЖЕНИЕ
# ============================================================================

if __name__ == "__main__":
    # Загрузка модели при запуске
    print("🚀 Инициализация приложения...")
    print("=" * 50)
    
    if load_model():
        # Создание интерфейса
        app = create_interface()
        
        print("=" * 50)
        print("✓ Приложение запущено успешно!")
        print(f"🌐 Откройте браузер и перейдите на локальный адрес")
        print("=" * 50)
        
        # Запуск приложения
        app.launch(share=False, debug=True, theme=gr.themes.Soft())
    else:
        print("❌ Невозможно запустить приложение без модели")
        print(f"   Убедитесь, что файл '{MODEL_PATH}' находится в текущей директории")
