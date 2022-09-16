from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from imutils.video import FPS
import cv2

# if we want to make it full screen 
from kivy.core.window import Window
# Window.fullscreen = 'auto'
# Window.borderless = True
videoFR = 30.0

class DemoCapture(MDApp):
    def build(self):
        #  Window.fullscreen = 'auto'
        layout = MDBoxLayout(orientation= 'vertical')
        self.image = Image()
        layout.add_widget(self.image)
        self.capture = cv2.VideoCapture('../backgroundVideos/2.mp4')
        Clock.schedule_interval(self.load_frame, 1.0/30.0)
        print("Video original FPS: " + str(self.capture.get(cv2.CAP_PROP_FPS)))
        videoFR = self.capture.get(cv2.CAP_PROP_FPS)
        Clock.schedule_interval(self.load_frame, 1.0/videoFR)
        return layout
    def load_frame(self, *args):
        ret, frame = self.capture.read()
        if not ret:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print('repeating')
            ret, frame = self.capture.read()
        self.image_frame = frame
        buffer = cv2.flip(frame,0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

def runVideoPlayer():
    d = DemoCapture()
    d.run()
