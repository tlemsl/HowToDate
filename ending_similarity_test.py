from sim import DateSimulation
import utill
import numpy as np


place, start_img= utill.reset_position() 
start_arr=np.array(start_img)
start_tuple=utill.numpy2tuple(start_arr)
env = DateSimulation(place, start_tuple)
env.get_image().show()
print(env._sim_checker.ending_check(env.get_state()))