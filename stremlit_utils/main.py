import streamlit as st
import time
from .config import *

st.set_page_config(page_title='lct_2024')

st.markdown('#### lct_2024')
st.markdown('##### Решение команды :blue-background[Ступор мозговины]')

with st.expander('**Инструкция**', expanded=True):
    st.info("""
    1. Выбрать тип ввода
    2. Загрузить файл(ы) 
    3. 
    """)

photo_tab, video_tab = st.tabs(['Загрузка фото', 'Загрузка видео'])

# model init

# чекать формат видео и фото

with photo_tab:
    input_container = st.container(border=True)

    photo_input = input_container.file_uploader('**Загрузить фото:**', accept_multiple_files=True)
    # проверка битых файлов?

    if len(photo_input) > 0:

        if input_container.button('Начать расчёт'):
            with input_container.status('Производится расчёт', expanded=True) as status:
                st.write('1. Считывание фотографий')
                time.sleep(1)
                
                st.write('2. Работа модели')
                #time.sleep(1)
                for file in photo_input:
                    pass
                    #st.text(file)
                    # model infer
                    # return
                    # 1 - ms/img (float)
                # из конфига подтянуть путь, куда сохран предск и показать, что загрузка туда зевершена (SAVE_ROOT)
                st.write('3. Вывод результатов')
                time.sleep(1)

            status.update(label='Расчёт окончен', state='complete', expanded=False)

with video_tab:
    input_container = st.container(border=True)

    video_input = input_container.file_uploader('**Загрузить видео:**')

    if video_input is not None:
        # проверка названия видео, если такое уже есть в логах, то подгружаем результ оттуда

        # если нет, то запуск раскадровки, инференс каждого н кадра + апроксим маежду ними + сохранение лога 
        # 

        # ms/img (float) + сколько на все видео
        # из конфига подтянуть путь, куда сохран предск и показать, что загрузка туда зевершена (SAVE_ROOT)
        
        # выбор классов, которые хотим отрисовать (мульти)
        
        # и выбранные классы идут в функц отрисовки
        # отрисовка результата (или для видео, кот уже есть, или для нового предск)

        # модель выдает:
        # время работы (для метрики) (флоат)
        # секунда(ы), где началась детекция (лист инт-флоат)
        # классы детекций (уже опасность или обычное размечено!!!!) для этих секунд (лист тегов)
        # добавить список приоритетных классов (бпла - опасные детекции, сам, верт, птица - обычные детекции)
        # список поямнить в сноске (и как раб ползунок)
        # ползунок
        # старт - опасность - прочее - прочее - ... - 

        type_1_container = st.container(border=True)
        type_1_container.markdown('**1. Выбор слайдером секунды**') # всегда доступен подный интервал!
        # мб выдвать что юоксов нет
        # 
        select_time = type_1_container.select_slider('Выбрать интервал (сек)', [0, 5, 10, 15, 20, 25, 30])

        type_1_container.video(video_input, start_time=select_time, end_time=select_time + 5,
                 autoplay=True, loop=False)

        """type_2_container = st.container(border=True)

        type_2_container.markdown('**2. Выбор списком**')
        moment_dict = {'Начало': 0,
                       'Первый интервал (потягивается)': 2,
                       'Второй интервал (чешет зад)': 10,
                       'Третий интервал (нюхает цветочки)': 20}

        select_moment = type_2_container.selectbox('Выбрать момент',
                                     moment_dict.keys())

        type_2_container.video(video_input, start_time=moment_dict[select_moment],
                 end_time=moment_dict[select_moment] + 5,
                 autoplay=True, loop=False)
        """

        # to update file limit up to 1.1g












