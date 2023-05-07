import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from tkinter import PhotoImage
import numpy

#load the trained model to classify sign
from keras.models import load_model
model = load_model('traffic_classifier.h5')

#dictionary to label all traffic signs class.
classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',      
            3:'Speed limit (50km/h)',       
            4:'Speed limit (60km/h)',      
            5:'Speed limit (70km/h)',    
            6:'Speed limit (80km/h)',      
            7:'End of speed limit (80km/h)',     
            8:'Speed limit (100km/h)',    
            9:'Speed limit (120km/h)',     
           10:'No passing',   
           11:'No passing veh over 3.5 tons',     
           12:'Right-of-way at intersection',     
           13:'Priority road',    
           14:'Yield',     
           15:'Stop',       
           16:'No vehicles',       
           17:'Veh > 3.5 tons prohibited',       
           18:'No entry',       
           19:'General caution',     
           20:'Dangerous curve left',      
           21:'Dangerous curve right',   
           22:'Double curve',      
           23:'Bumpy road',     
           24:'Slippery road',       
           25:'Road narrows on the right',  
           26:'Road work',    
           27:'Traffic signals',      
           28:'Pedestrians',     
           29:'Children crossing',     
           30:'Bicycles crossing',       
           31:'Beware of ice/snow',
           32:'Wild animals crossing',      
           33:'End speed + passing limits',      
           34:'Turn right ahead',     
           35:'Turn left ahead',       
           36:'Ahead only',      
           37:'Go straight or right',      
           38:'Go straight or left',      
           39:'Keep right',     
           40:'Keep left',      
           41:'Roundabout mandatory',     
           42:'End of no passing',      
           43:'End no passing veh > 3.5 tons' }
                 
#initialise GUI
top=tk.Tk()

top.geometry('800x600')
top.title('Traffic sign classification')
top.configure(background='#78A995')

from PIL import Image, ImageTk
img = Image.open("trafficimage.jpg")
img = img.resize((1, 1), Image.ANTIALIAS)
background_image = ImageTk.PhotoImage(file="newtraffic.jpg")
label1 = Label(top, image = background_image)
label1.place(x = 0,y = 0, relwidth=1.0, relheight=1.0)

heading = Label(top, text="ROAD SIGN DETECTION",pady=20, font=('arial',20,'bold'))
heading.configure(background='#C9CAFF',foreground='#364156')
heading.place(relx=0.5, rely=0.05, anchor=CENTER)

sign_image = Label(top)
sign_image.place(relx=0.5, rely=0.5, anchor=CENTER)

label=Label(top,background='#C8CAFD', font=('arial',15,'bold'))
label.place(relx=0.5, rely=0.8, anchor=CENTER)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((30,30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    print(image.shape)
    pred = model.predict([image])[0]
    pred_label = numpy.argmax(pred)
    pred_class = numpy.argmax(pred)
    sign = classes[pred_class+1]
    print(sign)
    label.configure(foreground='#011638', text=sign) 

def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='black', font=('arial',10,'bold'))
    classify_b.place(relx=0.5, rely=0.7, anchor=CENTER)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)

    except:
        pass

upload=Button(top,text="UPLOAD A SIGN",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',14,'bold'))
upload.place(relx=0.5, rely=0.9, anchor=CENTER)

top.mainloop()

from PIL import Image, ImageTk
img = Image.open("trafficimage.jpg")
img = img.resize((1, 1), Image.ANTIALIAS)
background_image = ImageTk.PhotoImage(file="newtraffic.jpg")
label1 = Label(top, image = background_image)
label1.place(x = 0,y = 0, relwidth=1.0, relheight=1.0)



upload=Button(top,text="UPLOAD A SIGN",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',14,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="ROAD SIGN DETECTION",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CEBAFF',foreground='#364156')
heading.pack()
top.mainloop()


'''
# Create a label for the background image
background_label = tk.Label(root, image=background_image)

# Place the label in the top-left corner of the window, spanning the whole window
background_label.place(x=0, y=0, relwidth=1, relheight=1)

'''
