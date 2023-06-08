from fastai.vision.all import *
import gradio as gr

learn_inf = load_learner('export.pkl')

categories = ('Car', 'Truck')

def classify_image(img):
  pred,pred_idx,probs = learn_inf.predict(img)
  return dict(zip(categories, map(float,probs)))

image = gr.inputs.Image(shape=(224,224))
label = gr.outputs.Label()
examples = ['truck.jpg', 'car.jpg', 'plane.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch()

