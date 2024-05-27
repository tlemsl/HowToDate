import pyautogui
import time
from PIL import ImageGrab

pyautogui.FAILSAFE = False
"""
resolution
720x405
960x540
1024x576
1280x720
1600x900
1920x1080

action
(x, y, type)
    type:
        c: click
        d: drag
        w: wheel
""" 
def cosine_similarity(img1, img2, resize_shape=(10,10)):
    array1 = np.array(img1.resize(resize_shape))
    array2 = np.array(img2.resize(resize_shape))
    assert array1.shape == array2.shape
    
    h, w, c = array1.shape
    len_vec = h * w * c
    vector_1 = array1.reshape(len_vec,) / 255.
    vector_2 = array2.reshape(len_vec,) / 255.

    cosine_similarity = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
    return cosine_similarity


class DateSimulation:
    def __init__(self, resolution = "1920x1080", start_pixel = (0,0), scale=100) -> None:
        self._start_x, self._start_y = start_pixel
        self._width, self._height = map(int, resolution.split('x'))
        self._scale = scale
    
    def _command_to_pixel(self, x, y):
        return x//self._scale*self._width + self._start_x, y//self._scale*self._height + self._start_y

    def get_image(self):
        return ImageGrab.grab((self._start_x, self._start_y, self._start_x+self._width, self._start_y + self._height))
    
    def get_state(self):
        return self._wait_for_stabilized()
    
    def step(self, action):
        x, y, type = action
        x, y = self._command_to_pixel(x,y)
        if type=="c":
            pyautogui.moveTo(x,y)
            pyautogui.click()
        elif type == "d":
            pyautogui.dragTo(x,y)
        
        return self.get_state()

    def _wait_for_stabilized(self):
        prev_state = self.get_image()
        while True:
            current_state = self.get_image()
            if np.all(np.abs(np.array(current_state) - np.array(prev_state)) < 1):
                return current_state
            self._try_to_skip()
            prev_state = current_state
            time.sleep(0.5)
    
    def _try_to_skip(self):
        pyautogui.press("space")

    def get_reward(self):
        return 0.0, False

    def reset(self):
        pass

