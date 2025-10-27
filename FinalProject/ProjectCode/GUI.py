from tkinter import colorchooser
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
from torchvision import transforms
import torch
from torchvision import models, datasets

# class DrawingApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("GUI")
#         self.root.geometry("1000x500")  # Adjust height if needed

#         # Set up left canvas for drawing
#         self.left_canvas = tk.Canvas(root, bg="white", width=400, height=400)
#         self.left_canvas.pack(side="left", padx=10, pady=(10, 0))  # Adjust padding for better alignment

#         # Create an Image and Draw object to save drawings
#         self.image = Image.new("RGB", (400, 400), "white")
#         self.draw = ImageDraw.Draw(self.image)
        
#         # Bind drawing events to the left canvas
#         self.brush_size = 5
#         self.left_canvas.bind("<B1-Motion>", self.paint)
#         self.left_canvas.bind("<ButtonRelease-1>", self.reset)  # Reset when the button is released
#         self.brush_color = "black"
#         self.last_x, self.last_y = None, None  # Track last position for smooth drawing

#         # Create a frame for buttons under the left canvas
#         button_frame = tk.Frame(root)
#         button_frame.pack(side="left", padx=10, pady=10, anchor="n")  # Place under the canvas

#         color_button = tk.Button(button_frame, text="Choose Color", command=self.choose_color)
#         color_button.pack(side="top", fill="x", pady=2)

#         clear_button = tk.Button(button_frame, text="Clear Canvas", command=self.clear_canvas)
#         clear_button.pack(side="top", fill="x", pady=2)

#         save_button = tk.Button(button_frame, text="Save Drawing", command=self.save_drawing)
#         save_button.pack(side="top", fill="x", pady=2)

#         predict_button = tk.Button(button_frame, text="Predict", command=self.predict)
#         predict_button.pack(side="top", fill="x", pady=2)

#         # Set up right canvas for image display
#         self.right_canvas = tk.Canvas(root, width=400, height=400)
#         self.right_canvas.pack(side="left", padx=10, pady=10)
        
#         # Load and display image on the right canvas
#         self.display_image("test.jpeg")  # Replace with your image path

#         # Add a static text label to the GUI
#         self.text_label = tk.Label(root, text="Prediction Image", font=("Arial", 14), fg="white")
#         self.text_label.place(x=750, y=10)  # Adjust the position as needed

#     def choose_color(self):
#         color = colorchooser.askcolor()[1]
#         if color:
#             self.brush_color = color

#     def paint(self, event):
#         # Draw a line from the last position to the current position on both canvases
#         if self.last_x and self.last_y:
#             # Draw on the left Tkinter canvas
#             self.left_canvas.create_line(
#                 self.last_x, self.last_y, event.x, event.y,
#                 fill=self.brush_color, width=self.brush_size
#             )
            
#             # Draw on the PIL image to save later
#             self.draw.line(
#                 [self.last_x, self.last_y, event.x, event.y],
#                 fill=self.brush_color, width=self.brush_size
#             )

#         # Update the last position
#         self.last_x, self.last_y = event.x, event.y

#     def clear_canvas(self):
#         # Clear all drawings on the left canvas and the PIL image
#         self.left_canvas.delete("all")
#         self.image = Image.new("RGB", (400, 400), "white")  # Reset the image
#         self.draw = ImageDraw.Draw(self.image)

#     def save_drawing(self):
#         # Save the image to a file
#         file_path = "output.png"
#         self.image.save(file_path)
#         print(f"Drawing saved as {file_path}")

#     def predict(self):
#         # Logic for prediction
#         # 1. Save the image
#         self.save_drawing()
       
#         # 2. Load the trained model and dataset
#         # Load dataset

#         # Allocate path to the dataset
#         path = '/Users/patrickstar/Desktop/courses/ASU/EEE598_DeepLearning/FinalProject/Logo-2K+_WithoutCategory'
#         # Define transformer
#         transform = transforms.Compose([
#             transforms.Grayscale(num_output_channels=3),
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#         dataset = datasets.ImageFolder(path, transform=transform)

#         # Load model
#         model = models.alexnet(pretrained=None)
#         alex_path = '/Users/patrickstar/Desktop/courses/ASU/EEE598_DeepLearning/FinalProject/ProjectCode/AlexNet.pth'
#         state_dict = torch.load(alex_path)

#         model.load_state_dict(state_dict)
        
#         # 3. Make prediction based on the image and display the image
#         test_image_path = 'output.png'

#         image = Image.open(test_image_path)

#         image_tensor = transform(image).unsqueeze(0)


#         with torch.no_grad():
#             output = model(image_tensor)
#         _, predicted = torch.max(output, 1)

#         print(predicted[0])

#         for img, label in dataset:
#             if predicted == label:
#                 img_rgb = img.permute(1, 2, 0)
#                 self.display_image(img_rgb)
#                 break
    
#     def display_image(self, image_path):
#         image = Image.open(image_path)
#         image = image.resize((400, 400), Image.LANCZOS)
#         self.photo_image = ImageTk.PhotoImage(image)
#         self.right_canvas.create_image(0, 0, anchor="nw", image=self.photo_image)
    
#     def reset(self, event):
#         # Reset the last position when the mouse button is released
#         self.last_x, self.last_y = None, None


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GUI")
        self.root.geometry("1000x500")  # Adjust height if needed

        # Set up left canvas for drawing
        self.left_canvas = tk.Canvas(root, bg="white", width=400, height=400)
        self.left_canvas.pack(side="left", padx=10, pady=(10, 0))  # Adjust padding for better alignment

        # Create an Image and Draw object to save drawings
        self.image = Image.new("RGB", (400, 400), "white")
        self.draw = ImageDraw.Draw(self.image)
        
        # Bind drawing events to the left canvas
        self.brush_size = 5
        self.left_canvas.bind("<B1-Motion>", self.paint)
        self.left_canvas.bind("<ButtonRelease-1>", self.reset)  # Reset when the button is released
        self.brush_color = "black"
        self.last_x, self.last_y = None, None  # Track last position for smooth drawing

        # Create a frame for buttons under the left canvas
        button_frame = tk.Frame(root)
        button_frame.pack(side="left", padx=10, pady=10, anchor="n")  # Place under the canvas

        color_button = tk.Button(button_frame, text="Choose Color", command=self.choose_color)
        color_button.pack(side="top", fill="x", pady=2)

        clear_button = tk.Button(button_frame, text="Clear Canvas", command=self.clear_canvas)
        clear_button.pack(side="top", fill="x", pady=2)

        save_button = tk.Button(button_frame, text="Save Drawing", command=self.save_drawing)
        save_button.pack(side="top", fill="x", pady=2)

        predict_button = tk.Button(button_frame, text="Predict", command=self.predict)
        predict_button.pack(side="top", fill="x", pady=2)

        # Set up right canvas for image display
        self.right_canvas = tk.Canvas(root, width=400, height=400)
        self.right_canvas.pack(side="left", padx=10, pady=10)
        
        # Load and display image on the right canvas
        # self.display_image("test.jpeg")  # Replace with your image path

        # Add a static text label to the GUI
        self.text_label = tk.Label(root, text="Prediction Image", font=("Arial", 14), fg="white")
        self.text_label.place(x=750, y=10)  # Adjust the position as needed

    def choose_color(self):
        color = colorchooser.askcolor()[1]
        if color:
            self.brush_color = color

    def paint(self, event):
        if self.last_x and self.last_y:
            self.left_canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                fill=self.brush_color, width=self.brush_size
            )
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=self.brush_color, width=self.brush_size
            )
        self.last_x, self.last_y = event.x, event.y

    def clear_canvas(self):
        self.left_canvas.delete("all")
        self.image = Image.new("RGB", (400, 400), "white")
        self.draw = ImageDraw.Draw(self.image)

    def save_drawing(self):
        file_path = "output.png"
        self.image.save(file_path)
        print(f"Drawing saved as {file_path}")

    def predict(self):
        self.save_drawing()
       
        path = '/Users/patrickstar/Desktop/courses/ASU/EEE598_DeepLearning/FinalProject/Logo-2K+_WithoutCategory'
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = datasets.ImageFolder(path, transform=transform)

        model = models.alexnet(pretrained=False)
        alex_path = '/Users/patrickstar/Desktop/courses/ASU/EEE598_DeepLearning/FinalProject/ProjectCode/TrainedModels/AlexNet.pth'
        state_dict = torch.load(alex_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        
        test_image_path = 'output.png'
        image = Image.open(test_image_path)
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)
        _, predicted = torch.max(output, 1)

        # Display the predicted image from the dataset
        for img, label in dataset:
            if label == predicted.item():  # Ensure predicted class matches label
                img = transforms.ToPILImage()(img)  # Convert tensor to PIL Image
                self.display_image_from_pil(img)
                break

    def display_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((400, 400), Image.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(image)
        self.right_canvas.create_image(0, 0, anchor="nw", image=self.photo_image)

    def display_image_from_pil(self, image):
        # Resizes and displays a PIL image directly on the right canvas
        image = image.resize((400, 400), Image.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(image)
        self.right_canvas.create_image(0, 0, anchor="nw", image=self.photo_image)

    def reset(self, event):
        self.last_x, self.last_y = None, None

# Run the application
root = tk.Tk()
app = DrawingApp(root)
root.mainloop()