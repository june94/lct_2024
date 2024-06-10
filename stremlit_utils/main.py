import streamlit as st
import time
#from .config import *

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

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

with photo_tab:
    input_container = st.container(border=True)

    photo_input = input_container.file_uploader('**Загрузить фото:**', accept_multiple_files=True,
                                                type=['png', 'jpg'])
    # проверка битых файлов?

    if len(photo_input) > 0:

        if input_container.button('Начать расчёт', key='photo_upload'):
            with input_container.status('Производится расчёт', expanded=True) as status:
                st.write('1. Считывание фотографий')
                time.sleep(1)
                
                st.write('2. Работа модели')
                #time.sleep(1)
                for file in photo_input:
                    #pass
                    st.text(file.name)
                    st.text(type(file))
                    st.image(file)
                    #st.text(file)
                    # model infer
                    # return
                    # 1 - ms/img (float)
                # из конфига подтянуть путь, куда сохран предск и показать, что загрузка туда зевершена (SAVE_ROOT)
                st.write('3. Вывод результатов')
                time.sleep(1)

            status.update(label='Расчёт окончен. Предсказания сохранены в *SAVE_ROOT*', state='complete', expanded=False)

with video_tab:
    input_container = st.container(border=True)

    box_found = input_container.checkbox('Найдены боксы? (временно)', value=True)
    time_elapsed = input_container.number_input('Сколько времени затратило? (временно)')

    video_input = input_container.file_uploader('**Загрузить видео:**',
                                                type=['mp4', 'mkv', 'mov', 'avi'])

    if video_input is not None:
        st.text(type(video_input))

        if input_container.button('Начать расчёт', key='video_upload'):
            st.session_state['button_clicked'] = True

            with input_container.status('Производится расчёт', expanded=True) as status:
                st.write('1. Проверка названия видео')
                # проверка названия видео, если такое уже есть в логах, то подгружаем результ оттуда
                # если нет, то запуск раскадровки, инференс каждого н кадра + апроксим маежду ними + сохранение лога

                time.sleep(1)
                st.write('2. Работа модели')
                time.sleep(2)
                # ms/img (float) + сколько на все видео
                # из конфига подтянуть путь, куда сохран предск и показать, что загрузка туда зевершена (SAVE_ROOT)
                st.write('3. Отрисовка результата')
                time.sleep(2)

            st.session_state.video_output = video_input

            status.update(label='Расчёт окончен', state='complete', expanded=False)

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

        if st.session_state['button_clicked']:
            video_output = st.session_state.video_output
            result_container = st.container(border=True)
            result_container.markdown('##### Вывод результатов')
            # всегда доступен подный интервал!

            result_container.markdown('Времени затрачено: ' + str(time_elapsed))

            # мб выдвать что боксов нет
            if box_found:
                # выбор классов, которые хотим отрисовать (мульти)
                class_select = result_container.multiselect('Выбрать класс(ы) для отображения:',
                                                           ['0 - БПЛА коптерного типа',
                                                            '1 - самолет',
                                                            '2 - вертолет',
                                                            '3 - птица',
                                                            '4 - БПЛА самолетного типа'])

                moment_dict = {'Начало': 0,
                               'Прочее 1 (потягивается)': 2,
                               'Опасность 1 (чешет зад)': 10,
                               'Прочее 2 (нюхает цветочки)': 20}

                select_moment = result_container.select_slider('Выбрать интервал (сек)', moment_dict.keys())

                result_container.video(video_output, start_time=moment_dict[select_moment],
                     end_time=moment_dict[select_moment] + 5,
                     autoplay=True, loop=False)
            else:
                result_container.error('Боксов не найдено')
                result_container.video(video_output, autoplay=True, loop=False)













