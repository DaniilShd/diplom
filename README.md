# Разработка методологии быстрой адаптации детектора поверхностных дефектов, сформированного на основе фундаментальной модели компьютерного зрения, для интеграции в компактную архитектуру

Выполнил: Шайдуров Даниил Сергеевич

Образовательная программа: Компьютерное зрение и нейронные сети

Группа: 292405-1

## Структура проекта

```
diplom_final/
├── analysis/
│   ├── original/
│   │   ├── config.yaml
│   │   └── scripts/                 # Анализ текстуры, яркости, FFT, кластеризация
│   └── synthetic/
│       ├── config.yaml
│       └── scripts/                 # Сравнение синтетики и реальных данных (DINOv2)
├── experiments/
│   ├── config.yaml
│   ├── exp1_frozen/run.py
│   ├── exp2_finetune/run.py
│   ├── exp3_ssl/run_distillation.py
│   └── scripts/
│       ├── train_ltdetr.py          # Обучение LTDETR
│       ├── evaluate.py              # Оценка качества
│       └── statistical_analysis.py  # Статистический анализ
├── generate/
│   ├── config.yaml
│   ├── ablation/
│   │   ├── run_ablation.py          # Запуск сетки параметров
│   │   ├── generate_synthetic.py    # Генерация для одной комбинации
│   │   ├── train_ltdetr.py          # Обучение LTDETR в абляции
│   │   └── evaluate.py              # Оценка в абляции
│   ├── scripts/
│   │   ├── config.py                # Загрузка конфигурации
│   │   ├── main.py                  # PoissonDefectGenerator
│   │   ├── sd_generator.py          # SDDefectGenerator
│   │   ├── generate_dataset.py      # Пакетная генерация
│   │   └── visualize_bbox.py        # Визуализация разметки
│   └── utils/
│       ├── blending.py              # Многомасштабное смешивание
│       ├── color_correction.py      # Цветокоррекция в LAB
│       ├── spectral.py              # Спектральное согласование
│       ├── scaling.py               # Масштабирование дефектов
│       ├── rle_utils.py             # Декодирование RLE
│       └── io_utils.py              # Ввод-вывод
├── prepare_dataset/
│   ├── config.yaml
│   ├── scripts/
│   │   ├── run_prepare.py           # Точка входа
│   │   ├── augment_real.py          # Аугментация реальных данных
│   │   ├── augment_synthetic.py     # Аугментация синтетики
│   │   ├── copy_real.py             # Копирование реальных данных
│   │   ├── copy_synthetic.py        # Копирование синтетики
│   │   ├── merge_datasets.py        # Слияние датасетов
│   │   └── validate_resize.py       # Валидация размеров
│   └── utils/
│       ├── augmentation.py          # Функции аугментации
│       └── dataset_utils.py         # Утилиты датасета
├── processed/
│   ├── config.yaml
│   ├── scripts/
│   │   ├── 01_analyze_csv.py        # Анализ CSV с разметкой
│   │   ├── 02_extract_patches.py    # Нарезка патчей с дефектами
│   │   ├── 02b_extract_clean_patches.py  # Нарезка чистых патчей
│   │   ├── 03_analyze_patches.py    # Анализ патчей
│   │   ├── 04_balance_defect_split.py    # Балансировка дефектных патчей
│   │   ├── 04b_balance_clean_split.py    # Балансировка чистых патчей
│   │   ├── 05_visualize_bboxes.py   # Визуализация боксов
│   │   └── check_dataset.py         # Проверка целостности датасета
│   └── utils/
│       ├── rle_utils.py             # Декодирование RLE
│       ├── yolo_utils.py            # Конвертация в YOLO
│       ├── patch_utils.py           # Утилиты патчей
│       ├── clean_patch_utils.py     # Утилиты чистых патчей
│       ├── io_utils.py              # Ввод-вывод
│       ├── report_utils.py          # Генерация отчётов
│       └── visualization_utils.py   # Визуализация
├── distillation/
│   └── backbone_distillation/
│       ├── config.yaml
│       ├── 00_extract_clean_weights.py
│       ├── 01_pretrain_backbone.py
│       ├── 02_train_detectors.py
│       ├── 03_evaluate.py
│       ├── 04_pseudo_labeling_experiment.py
│       ├── run_full_experiment.sh
│       └── utils/
│           ├── dataset.py           # YOLODataset
│           └── backbone_loader.py   # Загрузка весов LightlyTrain
├── download_data/
│   ├── config.yaml
│   └── download_data.py
├── docker-compose.yml
├── Makefile
└── config.yaml
```


## Основные результаты

| Модель | mAP@50:95 | mAP@50 | mAP@75 | FPS | Параметры, М | Размер, МБ |
|---|---|---|---|---|---|---|
| Учитель LTDETR | 0.255 | 0.463 | 0.261 | 60.9 | 41.7 | 159.5 |
| Faster R-CNN (scratch) | 0.144 | 0.378 | 0.083 | 78.6 | 28.3 | 108.0 |
| Faster R-CNN (ImageNet) | 0.178 | 0.395 | 0.130 | 80.0 | 28.3 | 108.0 |
| Faster R-CNN (distilled) | 0.209 | 0.430 | 0.137 | 80.3 | 28.3 | 108.0 |

Дистилляция дала прирост mAP@50:95 с 0,178 до 0,209 относительно ImageNet-предобучения. Скорость инференса на CPU — около 80 FPS, размер модели — 108 МБ.

## Запуск

Все эксперименты выполняются в Docker-контейнерах:

## Запуск

Все эксперименты выполняются в Docker-контейнерах:

## Запуск

Все эксперименты выполняются в Docker-контейнерах:

```bash
make up                # Запуск всех сервисов

make shell-prepare_dataset # Подготовка и балансировка патчей
exit

make shell-generate # Генерация синтетического датасета
exit

make shell-experiments # Эксперименты главы 3
exit

make shell-distill # Дистилляция и обучение Faster R-CNN
exit

