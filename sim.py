import pyautogui
import time
from PIL import ImageGrab

pyautogui.FAILSAFE = False
"""
resolution
720x405
960x540
1024x720
1280x720
1600x900
1920x1080

action
(x, y, type)
    type:
        c: click
        d: drag
""" 
class DateSimulation:
    def __init__(self, resolution = "1920x1080", start_pixel = (0,0), scale=100) -> None:
        self._start_x, self._start_y = start_pixel
        self._width, self._height = map(int, resolution.split('x'))
        self._scale = scale
    
    def _command_to_pixel(self, x, y):
        return x/self._scale*self._width + self._start_x, y/self._scale*self._height + self._start_y

    def get_state(self):
        return ImageGrab.grab((self._start_x, self._start_y, self._start_x+self._width, self._start_y + self._height))
    
    def step(self, action):
        x, y, type = action
        x, y = self._command_to_pixel(x,y)
        if type=="c":
            pyautogui.moveTo(x,y)
            pyautogui.click()
        elif type == "d":
            pyautogui.dragTo(x,y)
        
        return self.get_state()


if __name__ == "__main__":
    sim = DateSimulation()
    sim.step((50,90,'c')).show()
    time.sleep(4)

    sim.step((60,90,'c')).show()
    time.sleep(4)
    
    sim.step((40,90,'c')).show()
    time.sleep(4)
    
    sim.step((50,90,'c')).show()
    time.sleep(4)
