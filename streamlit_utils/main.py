import streamlit as st
import time
import os

import sys
sys.path.append("/home/Документы/lct_2024/lct_2024/stremlit_utils")

from config import *
from model import Model
from image_utils import *
from video_utils import *

st.set_page_config(page_title='lct_2024')
st.markdown('#### Нейросеть для мониторинга воздушного пространства вокруг аэропортов')
st.markdown('##### ЛЦТ 2024. Решение команды :blue-background[Ступор мозговины]')

@st.cache_data
def load_model():
    # Load large model
    model = Model()
    return model

model = load_model()

with st.expander('**Инструкция**', expanded=True):
    st.info("""
    1. Выбрать тип ввода (фото или видео)
    2. Загрузить файл(ы) 
    """)

photo_tab, video_tab = st.tabs(['Загрузка фото', 'Загрузка видео'])

if 'slider_button_clicked' not in st.session_state:
    st.session_state.slider_button_clicked = False
if 'video_button_clicked' not in st.session_state:
    st.session_state.video_button_clicked = False
if 'choice_button_clicked' not in st.session_state:
    st.session_state.choice_button_clicked = False

with photo_tab:
    input_container = st.container(border=True)

    photo_input = input_container.file_uploader('**Загрузить фото:**', accept_multiple_files=True,
                                                type=['png', 'jpg'])

    if len(photo_input) > 0:

        if input_container.button('Начать расчёт', key='photo_upload'):
            with input_container.status('Производится расчёт', expanded=True) as status:

                avg_time = 0
                for file in photo_input:
                    image = read_image(file)
                    start = time.time()
                    result = model(image)
                    avg_time +=  time.time() - start
                    if result is None:
                        input_container.error(f"Проблема с файлом {file.name}")
                        continue
                    model.save_preds(result, image.shape[0], image.shape[1], file.name, save_path=SAVE_IMAGES_TXT)
                avg_time = np.round(avg_time/len(photo_input), 4) 

                status.update(label=f'Расчёт окончен. Предсказания сохранены в {SAVE_IMAGES_TXT}. Время инференса модели {avg_time} мс/фото', state='complete', expanded=False) 

with video_tab:
    input_container = st.container(border=True)

    video_input = input_container.file_uploader('**Загрузить видео:**',
                                                type=['mp4', 'mkv', 'mov', 'avi'])

    if video_input is not None:
        video = Video(video_input)

        if input_container.button('Начать расчёт', key='video_upload'):
            st.session_state['video_button_clicked'] = True
            
            with input_container.status('Работа с видео', expanded=True) as status:
                st.write('1. Поиск видео в логах')
                
                if len(os.listdir(video.labels_path)):
                    st.write('2. Видео найдено!')
                else:
                    st.write('2. Видео не найдено, инференс модели')
                    avg_time = 0
                    image_dir = os.listdir(video.init_images_path)
                    for image in image_dir:
                        full_image_path = f"{video.init_images_path}/{image}"
                        start = time.time()
                        result = model(full_image_path)
                        avg_time += time.time() - start
                        model.save_preds(result, video.height, video.width, image, save_path=video.labels_path, save_thresh=THRESH)
                    avg_time = np.round(avg_time/len(video.init_images_path), 4)
                    
                    input_container.success(f'Предсказания сохранены в {video.root_dir}. Время инференса модели {avg_time} мс/фото')

                status.update(label=f'3. Расчёт окончен', state='complete', expanded=False)
                    

        if st.session_state['video_button_clicked']:
            result_container = st.container(border=True)
            result_container.markdown('##### Вывод результатов')

            select_classes = result_container.multiselect('Выбрать класс(ы) для отображения (если не выбрать ничего - отразятся все предсказания):',
                                                           CATEGORIES,)
            if len(select_classes):
                select_classes = [CATEGORIES.index(i) for i in select_classes]
            else:
                select_classes = list(range(len(CATEGORIES)))

            if result_container.button('Создать видео'):
                st.session_state['choice_button_clicked'] = True
            
                timestamps = video.make_video(select_classes)
                video_output = video.res_path

                st.session_state['timestamps'] = timestamps
                st.session_state['video_output'] = video_output


            # TODO подписать ползунок - пояснить
            if st.session_state['choice_button_clicked']:

                if len(st.session_state['timestamps']) > 1:
                    #select_moment = result_container.select_slider('Выбрать интервал', st.session_state['timestamps'].keys())
                    select_moment = result_container.selectbox('Выбрать интервал', st.session_state['timestamps'].keys())
                    
                    st.session_state['slider_button_clicked'] = True

                    result_container.video(st.session_state['video_output'], start_time=st.session_state['timestamps'][select_moment],
                        autoplay=True, loop=False)
                    
                
                else:
                    result_container.error('Боксов не найдено')
                    result_container.video(st.session_state['video_output'], autoplay=True, loop=False)

    else: 
        st.session_state['video_output'] = False
        st.session_state['timestamps'] = False
        st.session_state['slider_button_clicked'] = False
        st.session_state['choice_button_clicked'] = False
        st.session_state['video_button_clicked'] = False















