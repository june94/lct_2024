import os
from copy import deepcopy
from tarfile import ReadError

import streamlit as st

from streamlit_utils.config import *
from streamlit_utils.model import Model
from streamlit_utils.image_utils import *
from streamlit_utils.video_utils import *

@st.cache_resource 
def load_model() -> Model:
    """Функция-обертка, сеециально для Stremlit, предотвращяет повторную инициализацию модели в результате любый действий на странице.

    Returns:
        Model: модель детекции
    """
    model = Model(MODEL_PATH)
    return model


st.set_page_config(page_title='lct_2024')
st.markdown('#### Нейросеть для мониторинга воздушного пространства вокруг аэропортов')
st.markdown('##### ЛЦТ 2024. Решение команды :blue-background[Ступор мозговины]')

model = load_model()

with st.expander('**Инструкция**', expanded=True):
    st.info("""
    1. Выбрать тип ввода (фотографии или видео)
    2. Загрузить файл(ы),  
    3. Дождаться загрузки данных, нажать "Начать расчёт"
    
    В случае работы с видео:\n
    4. Выбрать классы, которые хотите увидеть на видео (опционально)
    5. Нажать "Создать видео"
    6. Если ббоксы (выбранных классов) найдены, в выпадающем списке можно выбрать интересующие для просмотра фрагменты вида *тэг*_*индекс*, где тег *ОПАСНОСТЬ* относится ко всем видам БПЛА, а *Прочее* - к птицам, вертолетам и самолетам
            
    Рекомендации:\n
    - Если количество фото >100, рекомендуется загружать их не поштучно, а в tar-архиве
    - Для загрузки нового видео желательно обновлять страницу (актуально для больших файлов)
    - При работе на машине с RAM<16GB при формировании видео на этапе 5 Streamlit может упасть, но само видео можно будет найти в логах (актуально для больших файлов)
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

    photo_input = input_container.file_uploader('**Загрузить фото или архив:**', 
                                                accept_multiple_files=True,
                                                type=['png', 'jpg', 'tar'])

    if len(photo_input) > 0:
        if input_container.button('Начать расчёт', 
                                  key='photo_upload'):

            # check if tar is loaded and process it
            if photo_input[0].name.endswith(".tar"):
                try:
                    archive = ImageTar(photo_input[0])
                    proc_input = archive.get_paths()
                except ReadError:
                    input_container.error(f"Проблема с архивом {photo_input[0].name}")
                    raise ReadError("Tar is not valid!")
            else:
                archive = None
                proc_input = deepcopy(photo_input)
                
            # sort data by size for batch inference optimization
            proc_input.sort(key=lambda x: x.size)

            with input_container.status('Производится расчёт', expanded=True) as status:
                avg_time = 0
                my_bar = status.progress(0)
                
                batch_data, huge_imgs = [], []
                batch_meta, huge_meta = [], []
                
                for c, file in enumerate(proc_input):
                    image = read_image(file)  
                    if image is None:
                        input_container.error(f"Проблема с файлом {file.name}")
                        continue

                    img_meta = [image.shape[0], image.shape[1], file.name]

                    if image.shape[0]*image.shape[1] < IMAGE_SIZE_THRESH: 
                        batch_data.append(image)
                        batch_meta.append(img_meta)
                    else:
                        huge_imgs.append(image)
                        huge_meta.append(img_meta)

                    if c%IMG_BATCH_SIZE == 0 or c == (len(proc_input)-1):

                        # batch inference with standard yolo model
                        if len(batch_data):
                            yolo_results = model(batch_data, slice_infer=False) #[0]
                            
                            for result, meta in zip(yolo_results, batch_meta):
                                avg_time += result.speed['inference']/1000
                                model.save_preds(result, 
                                                meta[0], 
                                                meta[1], 
                                                meta[2], 
                                                save_path=SAVE_IMAGES_TXT)

                        # 1 huge image inference with sahi
                        sahi_results = []
                        for huge_img, meta in zip(huge_imgs, huge_meta):
                            x_ratio, y_ratio, slice_width, slice_height = get_auto_slice_params(height=meta[0], width=meta[1])
                            sahi_results.append(model(huge_img, 
                                                    slice_infer=True, 
                                                    x_ratio=x_ratio, 
                                                    y_ratio=y_ratio, 
                                                    slice_width=slice_width,
                                                    slice_height=slice_height))

                        for result, meta in zip(sahi_results, huge_meta):
                            avg_time += result.durations_in_seconds["prediction"]
                            model.save_preds(result, 
                                            meta[0], 
                                            meta[1], 
                                            meta[2], 
                                            save_path=SAVE_IMAGES_TXT)

                        batch_data, huge_imgs = [], []
                        batch_meta, huge_meta = [], []

                        my_bar.progress(c/len(proc_input))
                
                avg_time = np.round(len(proc_input)/avg_time, 2) 

                status.update(label=f'Расчёт окончен. Предсказания сохранены в {os.path.abspath(SAVE_IMAGES_TXT)}. Время инференса модели {avg_time} fps.', state='complete', expanded=False) 
                
                # remove archive if tar loaded
                if archive is not None:
                    archive.remove_dir()

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
                
                # check if yolo detections already saved
                if len(os.listdir(video.labels_path)) == len(video):
                    st.write('2. Видео найдено!')
                else:
                    st.write('2. Видео не найдено, инференс модели')
                    my_bar = status.progress(0)
                    
                    avg_time = 0
                    image_dir = os.listdir(video.init_images_path)
                    image_dir.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
                    
                    # check if video resolution is huge
                    sahi_condition = video.height*video.width >= IMAGE_SIZE_THRESH
                    video_batch_size = VIDEO_BATCH_SIZE + 1 if sahi_condition else VIDEO_BATCH_SIZE
                    if sahi_condition:
                        x_ratio, y_ratio, slice_width, slice_height = get_auto_slice_params(height=video.height, width=video.width)

                    for c, image_batch in enumerate(Video.batch(image_dir, video_batch_size)):
                        full_image_paths = [f"{video.init_images_path}/{image}" for image in image_batch]
                        
                        # if video is huge each VIDEO_BATCH_SIZE + 1 frame is inferenced with sahi, else - only yolo inference takes place
                        if sahi_condition:
                            sahi_path = full_image_paths[0]
                            if len(full_image_paths) > 1:
                                yolo_paths = full_image_paths[1:]
                                image_names = image_batch[1:]
                            else:
                                yolo_paths = []
                        else:
                            yolo_paths = full_image_paths
                            image_names = image_batch
                        
                        if len(yolo_paths):
                            yolo_results = model(yolo_paths, slice_infer=False) #[0]

                            for result, image_name in zip(yolo_results, image_names):
                                avg_time += result.speed['inference']/1000
                                model.save_preds(result, 
                                                video.height, 
                                                video.width, 
                                                image_name=image_name, 
                                                save_path=video.labels_path, 
                                                save_thresh=THRESH)

                        if sahi_condition:
                            sahi_result = model(sahi_path, slice_infer=True, 
                                                    x_ratio=x_ratio, 
                                                    y_ratio=y_ratio, 
                                                    slice_width=slice_width,
                                                    slice_height=slice_height)
                            
                            avg_time += sahi_result.durations_in_seconds["prediction"]
                            model.save_preds(sahi_result, 
                                            video.height, 
                                            video.width, 
                                            image_name=image_batch[0], 
                                            save_path=video.labels_path, 
                                            save_thresh=THRESH)
                            
                        my_bar.progress(c*video_batch_size/len(video))

                    avg_time = np.round(len(video)/avg_time, 2)
                    
                    input_container.success(f'Предсказания сохранены в {os.path.abspath(video.root_dir)}. Время инференса модели {avg_time} fps.')

                status.update(label=f'3. Расчёт окончен', state='complete', expanded=False)
                    

        if st.session_state['video_button_clicked']:
            result_container = st.container(border=True)
            result_container.markdown('##### Вывод результатов')

            select_classes = result_container.multiselect('Выбрать класс(ы) для отображения:',
                                                        CATEGORIES,
                                                        placeholder="Выберите класс(ы) для отображения, по умолчанию выбраны ВСЕ")
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

            if st.session_state['choice_button_clicked']:

                if len(st.session_state['timestamps']) > 0:
                    select_moment = result_container.selectbox('Выбрать интервал', st.session_state['timestamps'].keys())
                    
                    st.session_state['slider_button_clicked'] = True

                    result_container.video(st.session_state['video_output'], start_time=st.session_state['timestamps'][select_moment],
                        autoplay=False, loop=False)
                else:
                    result_container.error('Боксов не найдено')
                    result_container.video(st.session_state['video_output'], autoplay=True, loop=False)

    else: 
        st.session_state['video_output'] = False
        st.session_state['timestamps'] = False
        st.session_state['slider_button_clicked'] = False
        st.session_state['choice_button_clicked'] = False
        st.session_state['video_button_clicked'] = False
