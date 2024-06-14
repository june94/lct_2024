# ЛЦТ 2024
## Задача "Нейросеть для мониторинга воздушного пространства вокруг аэропортов"

### Структура репозитория

- Основной код проекта находится в директори **streamlit_utils**, остальные связаны с подготовкой решения и приложены для ознакомления
- В **weights** лежат веса модели в различных форматах
- **train_utils** содержит скрипт тренировки модели, конфиги, логи экспериментов
- В **analysis_utils** лежат скрипты предобработки датасетов для обучения
- В **research** содержатся заметки по существующим на рынке решениям задачи, открытым датасетам и идеям

### Инструкция по сборке и запуску контейнера

1. Из корневой директории репозитория выполнить команду `docker build --build-arg USER_ID=$UID -t lct .`
2. Далее: `docker run --gpus all -it --rm --ipc=host -p 8501:8501 -v $(pwd):/lct_2024 lct`
3. По адресу http://localhost:8501/ снает доступно веб-приложение (его также можно будет открыть на некоторых мобильных устройствах, находящихся в этой же сети)
4. Результаты работы модели пишутся в *streamlit_utils/logs*

### Конвертация в TensorRT (Опционально)

Так как trt-модели являются "железозависимыми", для их использования нужно выполнить конвертацию модели на устройстве, где непосредственно будет проходить инференс модели. Данный пункт является необязательным, т.к. при инференсе на небольшой локальной машине (GeForce GTX 1650 Mobile, VRAM 4GB), прироста в скорости работы не наблюдалось, а на прочих устройствах тестирование не проводилось.

1. Убрать коментарий со [строчки](https://github.com/june94/lct_2024/blob/main/Dockerfile#L23) в Dockerfile, собрать контейнер
2. Изменить [CREATE_TENSORRT](https://github.com/june94/lct_2024/blob/main/streamlit_utils/config.py#L47) на True и задать [MAX_TRT_BATCH](https://github.com/june94/lct_2024/blob/main/streamlit_utils/config.py#L48)
3. При запуске контейнера сначала произойдет конвертация модели, далее она станет доступна для инференса
4. При последующих запусках контейнера на данном устройстве с trt-моделью желательно снова изменить [CREATE_TENSORRT](https://github.com/june94/lct_2024/blob/main/streamlit_utils/config.py#L47) на False, чтобы избежать повторной конвертации, при желании вернуться к тестированию pytorch-модели [MAX_TRT_BATCH](https://github.com/june94/lct_2024/blob/main/streamlit_utils/config.py#L48) необходимо так же установить = -1

### Вид приложения
![](streamlit_example.gif)

