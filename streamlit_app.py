import os
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2 
import gdown

# Import functions from the other files
from classification import predict_class
from segmentation import preprocess_image, predict_segment, dice_coef

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon = ":brain:",
    layout = 'wide',
    initial_sidebar_state = 'expanded'
)

page_bg_img = '''
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stAppViewContainer"] {
background: #444444;
background-size: cover;
opacity: 1;
}

[data-testid="stHeader"] {
background-color: rgba(0, 0, 0, 0);
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

@st.cache_resource()
def load_model_classification():
    '''Load the model for classification'''
    try:
        vgg16_model_id = "1umGzzekuHY6fQkGMpQrOfqRcDDM1pO6s" #ResNet50 -> Demo
        vgg16_model_path = "vgg16_model_g.h5"
        if not os.path.exists(vgg16_model_path):
            try:
                gdown.download('https://drive.google.com/uc?id=' + vgg16_model_id, vgg16_model_path, quiet=False) 
                # download_file_from_google_drive(vgg16_model_id, vgg16_model_path)
            except Exception as e:
                st.error(f'Error downloading model with gdown: {e}')
                return None
        vgg16_model = tf.keras.models.load_model(vgg16_model_path)
        return vgg16_model
    except:
        raise Exception("Error: Failed to load the VGG16 model")
    
# Load the U-Net model
@st.cache_resource()
def load_model_segmentation():
    '''Load the model for segmentation'''
    try:
        unet_model_id = "1aoabD5njPZOfslOAkm1o-7WfUkrQ4xxw"
        unet_model_path = "unet_model_g.h5"
        if not os.path.exists(unet_model_path):
            try:
                gdown.download('https://drive.google.com/uc?id=' + unet_model_id, unet_model_path, quiet=False)
                # download_file_from_google_drive(unet_model_id, unet_model_path)
            except Exception as e:
                st.error(f'Error downloading model with gdown: {e}')
                return None
        unet_model = tf.keras.models.load_model(unet_model_path, custom_objects={'dice_coef': dice_coef})
        return unet_model
    except:
        raise Exception("Error: Failed to load the U-Net model")

def process_image(uploaded_img, uploaded_mask=None, vgg16_model=None, unet_model=None):
    # Convert the file to an opencv image
    # note: streamlit does not support cv2.imread directly
    uploaded_img = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    decoded_img = cv2.imdecode(uploaded_img, 1)
    
    # Display the original image and ground truth mask
    with st.sidebar:
        st.image(decoded_img, channels="BGR", use_column_width=True, caption="Original Image")
        if uploaded_mask is not None:
            uploaded_mask = np.asarray(bytearray(uploaded_mask.read()), dtype=np.uint8)
            decoded_mask = cv2.imdecode(uploaded_mask, 1)
            st.image(decoded_mask, use_column_width=True, caption="Ground Truth Mask")

    # Classificationgs
    with st.spinner(text="Classifying the image..."):
        
        # Predict if tumor
        tumor_class = predict_class(decoded_img, vgg16_model)

    # # VGG16 Based
    # if tumor_class == 0:
    #     st.success("Meningioma Tumor Detected!")

    # elif tumor_class == 1:
    #     st.success("Glioma Tumor Detected!")

    # elif tumor_class == 2:
    #     st.success("Pituitary Tumor Detected!")

    # elif tumor_class == 3:
    #     st.balloons()
    #     st.success("No Tumor Detected, Stay Healthy!")
        
    # ResNet50 Based
    if tumor_class == 2: 
        st.success("No Tumor Detected, Stay Healthy!")

    else:
        # Segmentation
        with st.spinner(text="Preprocessing the image..."):

            # Read the image into a numpy array
            image = np.array(decoded_img)
            preprocessed = preprocess_image(image)

            # Normalize the image to [0,1] for visualization 
            # because there are negative values in the image after preprocessing
            min = preprocessed.min()
            max = preprocessed.max()
            normalized = (preprocessed - min) / (max - min)
            
        with st.spinner(text="Segmenting the image..."):

            # The model accept image without normalization
            # Add batch dimension for TensorFlow
            expanded = np.expand_dims(preprocessed, axis=0)

            # Predict the segmentation mask 
            segment = predict_segment(expanded, unet_model)

            # Post-processing: count the number of 1s in the segmentation mask
            tumor_area = np.count_nonzero(segment == 1)

            # Define a threshold for the minimum tumor area
            TUMOR_AREA_THRESHOLD = 50  # adjust this value based on your data

            if tumor_area < TUMOR_AREA_THRESHOLD:
                st.balloons()
                st.success("No Tumor Detected, Stay Healthy!")
                segment = np.zeros_like(segment)
            else:
                st.error(f"Tumor Detected!")
                if tumor_class == 0: 
                    st.error("Glioma Tumor Detected!")
                if tumor_class == 1:
                    st.error("Meningioma Tumor Detected!")
                if tumor_class == 3:
                    st.error("Pituitary Tumor Detected!")

            # Convert the segmentation mask to 0-255 scale for visualization
            segment = segment * 255

        col1, col2 = st.columns(2)
        col1.image(normalized, caption="Preprocessed Image", width=300)
        col2.image(segment, caption="Segmented Tumor Area", width=300)

def main():
    # load the models
    # note: model diambil dari URL Google drive karena ukuran file yang besar
    st.title("Brain Tumor Detection")

    with st.spinner('Downloading VGG16 Model...'):
        vgg16_model = load_model_classification()

    with st.spinner("Downloading U-Net Model..."):
        unet_model = load_model_segmentation()

    if "uploaded_img" not in st.session_state:
        st.session_state.uploaded_img = None

    if "uploaded_mask" not in st.session_state:
        st.session_state.uploaded_mask = None

    # Prompt for image upload
    uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="img_uploader")
    if uploaded_img is not None:
        st.session_state.uploaded_img = uploaded_img

    # Prompt for mask upload
    uploaded_mask = st.file_uploader("Choose a mask...", type=["jpg", "png", "jpeg"], key="mask_uploader")
    if uploaded_mask is not None:
        st.session_state.uploaded_mask = uploaded_mask

    if st.session_state.uploaded_img is not None:
        process_image(st.session_state.uploaded_img, st.session_state.uploaded_mask, vgg16_model, unet_model)
    else:
        st.warning("Please upload an image to continue")


if __name__ == "__main__":
    main()
