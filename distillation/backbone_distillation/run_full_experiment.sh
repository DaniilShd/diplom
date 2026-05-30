#!/usr/bin/env bash
set -e

echo "Запуск эксперимента"
echo ""

echo "Шаг 1/3: Дистилляция бэкбона"
python 01_pretrain_backbone.py
echo ""

echo "Шаг 2/3: Обучение детекторов"
python 02_train_detectors.py
echo ""

echo "Шаг 3/3: Оценка моделей"
python 03_evaluate.py
echo ""

echo "Эксперимент завершён"
echo "Результаты: outputs/results/evaluation.json"

if [ -f "outputs/results/evaluation.json" ]; then
    echo ""
    echo "Краткие результаты:"
    python -c "
import json
data = json.load(open('outputs/results/evaluation.json'))
for r in data['models']:
    print(f\"  {r['model']:<35} mAP50:95={r.get('map50_95',0):.4f}  FPS={r.get('fps',0):.1f}\")
"
fi