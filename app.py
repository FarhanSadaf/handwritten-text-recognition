import streamlit as st
from PIL import Image
import os
import string
import json
import cv2
import numpy as np
import errno

from text_segmentation import words, page

from htr.generator import Tokenizer, DataGenerator

from htr.network import puigcerver, ctc_loss, ctc_decode
from tensorflow.keras.models import Model

import numpy as np
from htr.preprocessing import preprocess, normalization, threshold


st.set_page_config(page_title='Offline HTR', page_icon=Image.open('./images/icon.png'))

# Change html/css styles
HTML = '''
    <style>
        .css-hi6a2p {
            padding: 1rem 1rem 1rem;
        }

        .stTextArea label {
            display: none;
        }

        .css-2trqyj {
            width: inherit;
            padding: 0.3rem 0.8rem;
            font-weight: 300;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        h1, h3 {
            text-align: center;
        }
    </style>
'''
st.markdown(HTML, unsafe_allow_html=True)  # Rendering HTML


@st.cache
def load_model(input_size, d_model, target_path):
    # Model
    inputs, outputs = puigcerver(input_size=input_size, d_model=d_model)
    model = Model(inputs=inputs, outputs=outputs)

    # Load weights from target_path
    if os.path.isfile(target_path):
        model.load_weights(target_path)
        print('===== Pre-trained weights loaded =====')
        return model

    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), target_path)


def draw_boxes(img, boxes):
    img = img.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)
    return img


def main():
    BASE_DIR = os.getcwd()
    # Parameters
    input_size = (1024, 128, 1)
    charset_eng = string.printable[:95]
    bangla_chars = json.load(open(os.path.join(BASE_DIR, 'htr', 'data', 'bangla-characters.json')))
    charset_ban = ''.join(bangla_chars) + string.punctuation + ' '
    max_text_len = 128

    # Filepaths
    output_path_eng = os.path.join(BASE_DIR, 'htr', 'data', 'output-english')
    target_path_eng = os.path.join(output_path_eng, 'checkpoint_weights_english1.hdf5')
    output_path_ban = os.path.join(BASE_DIR, 'htr', 'data', 'output-bangla')
    target_path_ban = os.path.join(output_path_ban, 'checkpoint_weights_bangla.hdf5')

    st.write('''
        # Handwritten Text Recognition
    ''')

    option = st.selectbox('Choose the language', ('English', 'Bangla'))

    uploaded_file = st.file_uploader('Upload image', type=['png', 'jpeg', 'jpg'])
    if uploaded_file is None:
        return

    # Read image from fileuploader
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Text segmentation
    crop = page.detection(img)
    boxes = words.detection(crop)
    lines = words.sort_words(boxes)
    crop = threshold(crop, block_size=13)    # 11 or 15 works best

    col1, col2 = st.beta_columns(2)
    col1.subheader('Uploaded image')
    col1.image(crop, use_column_width=True)

    if st.button('Convert to text'):
        col2.subheader('Text segmentation')
        col2.image(draw_boxes(crop, boxes), use_column_width=True)

        # Load model
        if option == 'English':
            tokenizer = Tokenizer(charset=charset_eng, max_text_len=max_text_len)
            model = load_model(input_size=input_size, d_model=tokenizer.vocab_size + 1, target_path=target_path_eng)
        elif option == 'Bangla':
            tokenizer = Tokenizer(charset=charset_ban, max_text_len=max_text_len)
            model = load_model(input_size=input_size, d_model=tokenizer.vocab_size + 1, target_path=target_path_ban)

        # Predict
        progress = st.progress(0)
        progress_value = 0
        output = ''

        for line in lines:
            imgs = []
            for x1, y1, x2, y2 in line:
                word = crop[y1:y2, x1:x2]
                imgs.append(preprocess(img=word, input_size=input_size))

            X = normalization(imgs)
            Y_pred = model.predict(X)
            predictions, probabilities = ctc_decode(Y_pred)
            y = [tokenizer.decode(y) for y in predictions]
            output += ' '.join(y) + '\n'

            progress_value += (1 / len(lines)) - 1e-7
            progress.progress(progress_value)

        # Output
        st.subheader('Converted text')
        st.text_area('', output, height=300)


if __name__ == "__main__":
    main()
