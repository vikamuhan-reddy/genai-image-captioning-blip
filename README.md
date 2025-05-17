## Prototype Development for Image Captioning Using the BLIP Model and Gradio Framework

### AIM:
To design and deploy a prototype application for image captioning by utilizing the BLIP image-captioning model and integrating it with the Gradio UI framework for user interaction and evaluation.

### PROBLEM STATEMENT:
Image captioning combines computer vision and NLP to generate descriptive text for images. Existing models can struggle with complex or varied visuals. This project develops a prototype using the BLIP model and a Gradio interface for real-time captioning and user evaluation.

### DESIGN STEPS:

#### STEP 1:
Collect and preprocess a labeled image-caption dataset suitable for training or fine-tuning image captioning models. Format the dataset to match the BLIP model’s input requirements.

#### STEP 2:
Select a pretrained BLIP model and configure the training or fine-tuning pipeline using appropriate deep learning frameworks. Train and evaluate the model on the prepared dataset using relevant metrics like BLEU or CIDEr scores.

#### STEP 3:
Develop a Python script to load the fine-tuned BLIP model and implement a caption generation function. Build a Gradio interface with image upload and caption display features. Deploy the application locally or on a cloud platform for user interaction and feedback.

### PROGRAM:
```py
import os
import io
import IPython.display
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']
# Helper functions
import requests, json

#Image-to-text endpoint
def get_completion(inputs, parameters=None, ENDPOINT_URL=os.environ['HF_API_ITT_BASE']):
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL,
                                headers=headers,
                                data=json.dumps(data))
    return json.loads(response.content.decode("utf-8"))
import gradio as gr 

def image_to_base64_str(pil_image):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()
    return str(base64.b64encode(byte_arr).decode('utf-8'))

def captioner(image):
    base64_image = image_to_base64_str(image)
    result = get_completion(base64_image)
    return result[0]['generated_text']

gr.close_all()
demo = gr.Interface(fn=captioner,
                    inputs=[gr.Image(label="Upload image", type="pil")],
                    outputs=[gr.Textbox(label="Caption")],
                    title="Image Captioning with BLIP",
                    description="Caption any image using the BLIP model",
                    allow_flagging="never",
                    examples=["christmas_dog.jpeg", "bird_flight.jpeg", "cow.jpeg"])

demo.launch(share=True, server_port=int(os.environ['PORT2']))
gr.close_all()
```

### OUTPUT:
![image](./Screen%20Shot%201947-02-27%20at%2011.19.48.png)


### RESULT:
A functional image captioning prototype using the BLIP model was developed, generating accurate captions for uploaded images. The Gradio interface enabled interactive testing, showcasing strong performance and potential for future enhancements.
