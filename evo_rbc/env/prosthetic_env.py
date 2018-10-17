import statistics as stats

from osim.env import ProstheticsEnv

from .eaenv import EAenv


class ProstheticEAEnv(EAenv, ProstheticsEnv):

    def __init__(self, seed=1, max_time_steps_qd=300, max_time_steps_task=2000, visualize=False,joint_error_margin=0.1):
        ProstheticsEnv.__init__(self, visualize=visualize)
        self.seed(seed)
        EAenv.__init__(self, max_time_steps_qd=max_time_steps_qd, max_time_steps_task=max_time_steps_task)
        self.mpi_worker_path = '../../mpi_worker/prosthetic_evaluation_worker.py'
        self.joint_error_margin = joint_error_margin

    # self.logger.debug("Created prosthetic environment")

    def evaluate_task_fitness(self, task_funtion, arbitrator_genome, visualise=False):
        raise NotImplementedError

    def evaluate_quality_diversity_fitness(self, qd_function, primitive_genome, visualise=False):
        return qd_function(primitive_genome, visualise)

    def qd_steady_runner(self, primitive_genome, visualise=False):
        """quality_diversity fitness function for a runner with steady velocity vector"""
        done = False
        self.reset()

        pelvis_kinematics = {"vx": [], "vy": [], "vz": [], "rz": []}

        initial_state_desc = self.get_state_desc()
        initial_position_x = initial_state_desc["body_pos"]["pelvis"][0]

        performance = 0.0

        self.logger.debug("Starting an evaluation for steady skeleton runner")

        for time_step in range(self.max_time_steps_qd):

            if (not done):
                action = []
                for muscle_index in range(self.action_space.shape[0]):
                    action.append(primitive_genome.control_function(muscle_index=muscle_index, time_step=time_step))
                observation, reward, done, info = self.step(action)
                if (visualise):
                    self.render()
                state_desc = self.get_state_desc()
                pelvis_velocity_vector = state_desc["body_vel"]["pelvis"]
                pelvis_kinematics["vx"].append(pelvis_velocity_vector[0])
                if time_step == 100:
                    pelvis_position_x = state_desc["body_pos"]["pelvis"][0]
                    if (pelvis_position_x < initial_position_x):
                        ### stop computation since going backwards
                        mean_velocity_x = stats.mean(pelvis_kinematics["vx"])
                        behavior = mean_velocity_x
                        self.logger.debug("Evaluation stopped since going backwards behavior " + str(behavior))
                        return (-1000.0, -1000.0)

                pelvis_position_vector_y = state_desc["body_pos"]["pelvis"][1]
                if (pelvis_position_vector_y < 0.75):
                    performance -= 1

                performance += 5 * (state_desc["body_pos"]["head"][0] - state_desc["body_pos"]["pelvis"][0])

                ### restrict knee and hip angles

                knee_rot_min = -1.7
                knee_rot_max = 0.1
                hip_rot_min = -0.6
                hip_rot_max = 0.6


                osim_model = self.osim_model
                model = osim_model.model
                state = osim_model.state
                '''
                ### KNEE - should be between 0 to around 72 degree (1.25 radian) Error margin - 0.1
                #  ..... mostly should be near 20 degree
                knee_right_rot = model.getStateVariableValue(state, "knee_r/knee_angle_r/value")
                knee_left_rot = model.getStateVariableValue(state, "knee_l/knee_angle_l/value")
                
                if(knee_left_rot > (knee_rot_max + self.joint_error_margin) or knee_left_rot < (knee_rot_min-self.joint_error_margin)):
                    self.logger.debug("Evaluation stopped since left knee angle outside permissible range "
                                      +str(knee_left_rot) + " at time step " + str(time_step))
                    return (-2000.0, -2000.0)
                if(knee_right_rot > (knee_rot_max + self.joint_error_margin) or knee_right_rot < (knee_rot_min-self.joint_error_margin)):
                    self.logger.debug("Evaluation stopped since right knee angle outside permissible range "
                                      + str(knee_right_rot) + " at time step " + str(time_step))
                    return (-2000.0, -2000.0)

                ### HIP - should be between -20 (-0.34) to around 25 degree (0.43 radian) Error margin - 0.1
                hip_left_rot = model.getStateVariableValue(state, "hip_l/hip_flexion_l/value")
                hip_right_rot = model.getStateVariableValue(state, "hip_r/hip_flexion_r/value")

                if(hip_left_rot > (hip_rot_max + self.joint_error_margin) or hip_left_rot < (hip_rot_min-self.joint_error_margin)):
                    self.logger.debug("Evaluation stopped since left hip angle outside permissible range "
                                      +str(hip_left_rot) + " at time step " + str(time_step))
                    return (-2000.0, -2000.0)

                if(hip_right_rot > (hip_rot_max + self.joint_error_margin) or hip_right_rot < (hip_rot_min-self.joint_error_margin)):
                    self.logger.debug("Evaluation stopped since right hip angle outside permissible range "
                                      + str(knee_right_rot)+ " at time step " + str(time_step))
                    return (-2000.0, -2000.0)
                '''

                ############ end of angle restrictions

        mean_velocity_x = stats.mean(pelvis_kinematics["vx"])
        behavior = mean_velocity_x
        ##penalise negative velocity
        if (behavior < 0):
            performance -= 50.0
        else:
            performance += len(pelvis_kinematics["vx"]) * ((mean_velocity_x ** 2) - (stats.stdev(pelvis_kinematics["vx"])**2) + 0.25)

        self.logger.debug("Evaluation finished with\nbehavior " + str(behavior) + "\nperformance " + str(performance) 
            + "\n survived for timesteps " + str(len(pelvis_kinematics["vx"])))

        return (behavior, performance)
