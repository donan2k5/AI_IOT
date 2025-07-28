import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np

# Load the trained model to classify signs
from keras.models import load_model
model = load_model('my_model.h5')

# IMPORTANT:
# The model was trained on 43 classes. Its output will be an index from 0 to 42.
# We need to map the original index that the model predicts to our new desired output.
# From your original 'classes' dictionary:
# - 'Stop' was class 15 (model index 14)
# - 'Turn right ahead' was class 34 (model index 33)
# - 'Turn left ahead' was class 35 (model index 34)

# New dictionary to map the model's output index to your desired string
target_classes = {
    14: "stop: 0",
    33: "right: 2",
    34: "left: 1"
}

# Initialise GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Nhận dạng biển báo giao thông')
top.configure(background='#ffffff')

label = Label(top, background='#ffffff', font=('arial', 15, 'bold'))
sign_image = Label(top)

def classify(file_path):
    image = Image.open(file_path).convert('RGB')
    image = image.resize((30, 30))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)

    # Predict class
    pred_index = np.argmax(model.predict(image)[0])

    # Check if the predicted index is one of the classes we want
    if pred_index in target_classes:
        sign = target_classes[pred_index]
    else:
        # If the sign is not 'Stop', 'Left', or 'Right', show a message
        sign = "Không phải biển báo yêu cầu"
    
    print(f"Model predicted index: {pred_index} -> Output: {sign}")
    label.configure(foreground='#011638', text=sign)

def show_classify_button(file_path):
    classify_b = Button(top, text="Nhận dạng", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#c71b20', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except Exception as e:
        print(f"Error: {e}")
        pass

upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#c71b20', foreground='white', font=('arial', 10, 'bold'))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)

heading = Label(top, text="Nhận dạng biển báo giao thông", pady=10, font=('arial', 20, 'bold'))
heading.configure(background='#ffffff', foreground='#364156')
heading.pack()

heading1 = Label(top, text="Môn Học: Cơ sở ứng dụng AI", pady=10, font=('arial', 20, 'bold'))
heading1.configure(background='#ffffff', foreground='#364156')
heading1.pack()

heading2 = Label(top, text="Danh sách thành viên nhóm", pady=5, font=('arial', 20, 'bold'))
heading2.configure(background='#ffffff', foreground='#364156')
heading2.pack()

heading3 = Label(top, text="Văn Huy Du MSSV: 20119205", pady=5, font=('arial', 20, 'bold'))
heading3.configure(background='#ffffff', foreground='#364156')
heading3.pack()

top.mainloop()