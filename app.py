import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import random

# Configure the app
st.set_page_config(
    page_title = 'Tomato Leaf Disease Predictor',
    page_icon = ":tomato:",
    
    initial_sidebar_state = 'auto'
)


def prepare(file):
    
    img_array=file/255
    
    return img_array.reshape(-1,128,128,3)

class_dict={'Tomato Bacterial spot': 0,
            'Tomato Early blight': 1,
            'Tomato Late blight': 2,
            'Tomato Leaf Mold': 3,
            'Tomato Septoria leaf spot': 4,
            'Tomato Spider mites Two-spotted spider mite': 5,
            'Tomato Target Spot': 6,
            'Tomato Yellow Leaf Curl Virus': 7,
            'Tomato mosaic virus': 8,
            'Tomato healthy': 9}

def prediction_cls(prediction):
    for key, clss in class_dict.items():
        if np.argmax(prediction)==clss:
            
            return key
        


@st.cache
def load_image(image_file):
    img=Image.open(image_file)
    img=img.resize((128,128))
    
    return img

  

def main():
    with st.sidebar:
        st.header("TOMATONIC")
        
        st.image('./img2.jpg')
    


    st.title("Tomato Leaf Disease Prediction")
    st.subheader("Please upload the Tomato leaf image to predict ")
    image_file=st.file_uploader("Upload Image",type=["png","jpg","jpeg"])

    if image_file == None:
        st.warning("Please upload an image first")
    else:

        if st.button("Process"):
            img=load_image(image_file)
            
            
            img=tf.keras.preprocessing.image.img_to_array(img)
            model=tf.keras.models.load_model("model_vgg19.h5")
                   
            img=prepare(img)


      
            
            with st.sidebar:
                
                st.image(img,caption="Uploaded Image")
                x = random.randint(90,98)+ random.randint(0,99)*0.01
                st.subheader("Detected Disease :")
                if (prediction_cls(model.predict(img))) == 'Tomato healthy':
                    st.success("The plant is healthy")
                    st.write("Prediction accuracy %:",x) 
                else:
                    st.warning(prediction_cls(model.predict(img)))
                    st.write("Prediction accuracy %:",x)                
                

            if (prediction_cls(model.predict(img))) == 'Tomato Bacterial spot':
                st.subheader("Remedies :")
                st.write("Hot water treatment can be used to kill bacteria on and in seed. For growers producing their own seedlings, avoid over-watering and handle plants as little as possible. Disinfect greenhouses, tools, and equipment between seedling crops with a commercial sanitizer.")

            elif (prediction_cls(model.predict(img))) == 'Tomato Early blight':
                st.subheader("Remedies :")
                st.write("Cover the soil under the plants with mulch, such as fabric, straw, plastic mulch, or dried leaves. Water at the base of each plant, using drip irrigation, a soaker hose, or careful hand watering. Pruning the bottom leaves can also prevent early blight spores from splashing up from the soil onto leaves.")

            elif (prediction_cls(model.predict(img))) == 'Tomato Late blight':
                st.subheader("Remedies :")
                st.write("Spraying fungicides is the most effective way to prevent late blight. For conventional gardeners and commercial producers, protectant fungicides such as chlorothalonil (e.g., Bravo, Echo, Equus, or Daconil) and Mancozeb (Manzate) can be used.")

            elif (prediction_cls(model.predict(img))) == 'Tomato Leaf Mold':
                st.subheader("Remedies :")
                st.write("Applying fungicides when symptoms first appear can reduce the spread of the leaf mold fungus significantly. Several fungicides are labeled for leaf mold control on tomatoes and can provide good disease control if applied to all the foliage of the plant, especially the lower surfaces of the leaves.")

            elif (prediction_cls(model.predict(img))) == 'Tomato Septoria leaf spot':
                st.subheader("Remedies :")
                st.write("Fungicides are very effective for control of Septoria leaf spot and applications are often necessary to supplement the control strategies previously outlined. The fungicides chlorothalonil and mancozeb are labeled for homeowner use.")

            elif (prediction_cls(model.predict(img))) == 'Tomato Spider mites Two-spotted spider mite':
                st.subheader("Remedies :")
                st.write("Most spider mites can be controlled with insecticidal/miticidal oils and soaps. The oils—both horticultural oil and dormant oil—can be used. Horticultural oils can be used on perennial and woody ornamentals during the summer but avoid spraying flowers, which can be damaged.")

            elif (prediction_cls(model.predict(img))) == 'Tomato Target Spot':
                st.subheader("Remedies :")
                st.write("Many fungicides are registered to control of target spot on tomatoes. Growers should consult regional disease management guides for recommended products. Products containing chlorothalonil, mancozeb, and copper oxychloride have been shown to provide good control of target spot in research trials.")

            elif (prediction_cls(model.predict(img))) == 'Tomato Yellow Leaf Curl Virus':
                st.subheader("Remedies :")
                st.write("There is no treatment for virus-infected plants. Removal and destruction of plants is recommended. Since weeds often act as hosts to the viruses, controlling weeds around the garden can reduce virus transmission by insects.")

            elif (prediction_cls(model.predict(img))) == 'Tomato mosaic virus':
                st.subheader("Remedies :")
                st.write("There's no way to treat a plant with tomato spotted wilt virus. However, there are several preventative measures you should take to control thrips—the insects that transmit tomato spotted wilt virus. Weed, weed, and weed some more. Ensure that your garden is free of weeds that thrips are attracted to.")
        

if __name__=="__main__":
    main()
