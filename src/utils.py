from tkinter import Frame
import cv2
import torch
import torchvision
# from torchvision.models import resnet50, ResNet50_Weights

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

frame = None

class utils:
    @staticmethod
    def load_model():
        # Load the DeepLab v3 model to system
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        print("Prediction happening on: " + device)
        ## try to switch to new verison for better prediction
        # model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")

        ##uncomment to improve fps
        # model = model.half()
        
        model.to(device).eval()
        return model
    
    @staticmethod
    def grab_frame(cap):
        # Given a video capture object, read frames from the same and convert it to RGB
        _, frame = cap.read()
        scale_percent = 100 # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def get_pred(img, model):
        # See if GPU is available and if yes, use it
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Define the standard transforms that need to be done at inference time
        imagenet_stats = [[0.485, 0.456, 0.406], [0.485, 0.456, 0.406]]
        preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(mean = imagenet_stats[0],
                                                                                      std  = imagenet_stats[1])])
        input_tensor = preprocess(img).unsqueeze(0)

        ##uncomment to improve fps
        # input_tensor = input_tensor.half()
        
        input_tensor = input_tensor.to(device)

        # Make the predictions for labels across the image
        with torch.no_grad():
            output = model(input_tensor)["out"][0]
            output = output.argmax(0)
        
        # Return the predictions
        return output.cpu().numpy()