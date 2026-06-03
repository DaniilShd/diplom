# Makefile для управления Docker-пайплайном

.PHONY: build up down shell generate insert validate viz all clean

# Сборка образа
build:
	docker compose build

# Запуск контейнера
up:
	docker compose up -d

# Остановка контейнера
down:
	docker compose down

# Интерактивная оболочка
shell-processed:
	docker exec -it processed /bin/bash

shell-analysis:
	docker exec -it analysis /bin/bash

shell-generate:
	docker exec -it generate /bin/bash

shell-prepare_dataset:
	docker exec -it prepare_dataset /bin/bash

shell-experiments:
	docker exec -it experiments /bin/bash

shell-up-mlflow:
	docker compose up -d mlflow

shell-distill:
	docker exec -it distillation /bin/bash

