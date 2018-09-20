from evo_rbc.genome.prosthetic_genome import ProstheticGenome
from evo_rbc.env.prosthetic_env import ProstheticEAEnv
import evo_rbc.main.utils as test_utils
import numpy as np

logger = test_utils.getLogger()
prosthetic_env = ProstheticEAEnv(visualize=True)
prosthetic_env.reset()
prosthetic_genome = ProstheticGenome()

import time

start = time.time()
for i in range(1):
    total_reward = 0.0
    prosthetic_genome.mutate(0.5)
    prosthetic_env.reset()
    for k in range(10):
        action = [0.0] * 19
        for j in range(19):
            #  prosthetic_genome.control_function(j, k)
            action[j] = 0
        obs, reward, done, info = prosthetic_env.step(action, project=False)
        total_reward += reward

        obs_rot = obs["body_pos_rot"]
        tibia_l = np.array(obs_rot["tibia_l"])
        femur_l = np.array(obs_rot["femur_l"])
        #print(femur_l)
        print(tibia_l - femur_l)

        osim_model = prosthetic_env.osim_model
        model = osim_model.model
        state = osim_model.state

        ########### KNEE - should be between 0 to around 72 degree (1.25 radian)
        #  ..... mostly should be near 20 degree

        model.setStateVariableValue(state, "knee_r/knee_angle_r/value", 0)
        model.setStateVariableValue(state, "knee_l/knee_angle_l/value", -1.25)

        ########### HIP - should be between -20 (-0.34) to around 25 degree (0.43 radian)
        model.setStateVariableValue(state, "hip_r/hip_flexion_r/value", 0.43)
        model.setStateVariableValue(state, "hip_l/hip_flexion_l/value", -0.34)


        time.sleep(0.5)
        model.assemble(state)

        osim_model.integrate()

        print(model.getStateVariableValue(state, "knee_l/knee_angle_l/value"))
        print(model.getStateVariableValue(state, "knee_r/knee_angle_r/value"))
        print(model.getStateVariableValue(state, "hip_l/hip_flexion_l/value"))
        print(model.getStateVariableValue(state, "hip_r/hip_flexion_r/value"))

        if (done):
            break
    print(total_reward, i, k)

end = time.time()
print(end - start, end - start > 60)
