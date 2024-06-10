import streamlit as st
import time
import os

from .config import *
from .model import Model
from .image_utils import *
from .video_utils import *

st.set_page_config(page_title='lct_2024')
st.markdown('#### lct_2024')
st.markdown('##### Решение команды :blue-background[Ступор мозговины]')

@st.cache_data
def load_model():
    # Load large model
    model = Model()
    return model

model = load_model()

with st.expander('**Инструкция**', expanded=True):
    st.info("""
    1. Выбрать тип ввода
    2. Загрузить файл(ы) 
    3. 
    """)

photo_tab, video_tab = st.tabs(['Загрузка фото', 'Загрузка видео'])

with photo_tab:
    input_container = st.container(border=True)

    photo_input = input_container.file_uploader('**Загрузить фото:**', accept_multiple_files=True)

    if len(photo_input) > 0:

        if input_container.button('Начать расчёт'):
            with input_container.status('Производится расчёт', expanded=True) as status:
                st.write('1. Работа модели')

                avg_time = 0
                for file in photo_input:
                    image = read_image(file)
                    start = time.time()
                    result = model(image)
                    avg_time += start - time.time()
                    model.save_preds(result, image.shape[0], image.shape[1], file.name, save_path=SAVE_IMAGES_TXT)
                avg_time = avg_time/len(photo_input) # ms/img

                st.write('2. Вывод результатов')
                # вывод времени + показать, что загрузка зевершена в (SAVE_IMAGES_TXT)

                status.update(label='Расчёт окончен', state='complete', expanded=False) ## attention

with video_tab:
    input_container = st.container(border=True)

    video_input = input_container.file_uploader('**Загрузить видео:**')

    if video_input is not None:
        video = Video(video_input)
        if os.path.splitext(video_input.name)[0] in os.listdir(SAVE_VIDEO):
            # выдать что видео найдено
            pass
        else:
            for file in os.listdir(video.init_images_path):
                start = time.time()
                result = model(image)
                avg_time += start - time.time()
                model.save_preds(result, video.height, video.width, file.name, save_path=video.labels_path)
            avg_time = avg_time/len(video.init_images_path) # ms/img
            # ms/img (float) + сколько на все видео
            # # из конфига подтянуть путь, куда сохран предск и показать, что загрузка туда зевершена (SAVE_ROOT)
        
        # выбор классов, которые хотим отрисовать (мульти)
        # и выбранные классы идут в функц отрисовки

        # пояснить что выбираем все, если пользователем не выбрано иное
        select_classes = [] #UPD
        timestamps, tags = video.make_video(select_classes) 

        # как раб ползунок пояснить
        #video.res_path - ЗДЕСЬ ФИНАЛЬНОЕ ВИДЕО ДЛЯ ПОКАЗА

        type_1_container = st.container(border=True)
        type_1_container.markdown('**1. Выбор слайдером секунды**') # всегда доступен подный интервал!
        select_time = type_1_container.select_slider('Выбрать интервал (сек)', [0, 5, 10, 15, 20, 25, 30])

        type_1_container.video(video_input, start_time=select_time, end_time=select_time + 5,
                 autoplay=True, loop=False)












