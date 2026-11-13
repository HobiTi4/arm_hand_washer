# arm_hand_washer

Роборука (UFactory xArm7) для миття вікон. Мова документації й коментарів — українська.

## Середовище
- Python venv: `.venv` (`pip install -r requirements-dev.txt`; `requirements.lock` — pinned versions for the Docker image, regenerate it when `requirements.txt` changes). Завжди запускай через `.venv/bin/python`.
- ROS 2 Humble + MoveIt 2 + xarm_ros2 живуть у Docker: `docker compose -f docker/compose.yaml run --rm ros`.
- Секрети (GitHub token) — у `.env`, не читати й не комітити.

## Команди
- Демо: `cd sim/xarm7_window_head/scripts && ../../../.venv/bin/python demo_wipe.py [--view|--rigid]`
- Перегенерувати моделі після зміни розмірів головки: `scripts/generate.py` (єдине джерело параметрів; `mjcf/` і `urdf/` не правити руками).
- Тести: `.venv/bin/pytest`. Лінт: `.venv/bin/ruff check .`

## Домовленості
- Сила притиску 20 Н, пружний тримач леза обовʼязковий (див. README симуляції).
- Вода в симуляції не моделюється; самоколізії руки вимкнені в MJCF — перевіряє планувальник.
- Перед підключенням до реальної руки: перевірити межі зони й швидкості, лише невеликі швидкості на першому запуску.
