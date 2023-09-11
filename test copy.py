import glob
import os
import sys
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Concatenate
import random
import cv2
import math
from threading import Thread
from keras.callbacks import TensorBoard

import time
from collections import deque
from tqdm import tqdm


try:
    sys.path.append(glob.glob('C:\CARLA_0.9.13\WindowsNoEditor\PythonAPI\carla\dist\carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Constantes
IM_WIDTH = 680
IM_HEIGHT = 420
N_ACTIONS = 4
BATCH_SIZE = 32
GAMMA = 0.9
EPSILON_START = 0.995
EPSILON_FINAL = 0.05
EPSILON_DECAY = 0.0002
MEMORY_CAPACITY = 2000
TRAIN_START = 1000
TRAIN_INTERVAL = 50
TARGET_UPDATE_INTERVAL = 100
EPISODE_MAX = 500
RADAR_RANGE = 100
FPS = 30
LOSS = None
# Fréquence de mise à jour du modèle cible
UPDATE_TARGET_FREQUENCY = 1000

# Valeur minimale d'epsilon
EPSILON_END = 0.05


SHOW_PREVIEW = True
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"

MEMORY_FRACTION = 0.4
MIN_REWARD = -200

EPISODES = 100

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95 ## 0.9975 99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


class CarWorld:
    # Se connecter au serveur CARLA et créer un monde

    def __init__(self):
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        while True :
            try : 
                blueprint_library = self.world.get_blueprint_library()
                vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
                spawn_point = random.choice(self.world.get_map().get_spawn_points())
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                self.actor_list.append(self.vehicle)
                break
            except RuntimeError:
                print("retry")
                pass

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        camera_bp.set_attribute("fov", f"110")
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda data: self.process_img(data))
        self.actor_list.append(self.camera)


        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute("range",f'{RADAR_RANGE}')
        radar_transform = carla.Transform(carla.Location(x=0.8, z=1.0))
        self.radar = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.vehicle)
        self.radar.listen(lambda data: self.process_radar(data))
        self.actor_list.append(self.radar)

        collision_sensor_bp = blueprint_library.find('sensor.other.collision')

        # Créer un capteur de collision
        self.collision_sensor = self.world.spawn_actor(collision_sensor_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.collision_data(event))
        self.actor_list.append(self.collision_sensor)

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        while self.front_camera is None or self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(2)


    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self,image):
        i = np.array(image.raw_data)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        cv2.imshow("", i3)
        cv2.waitKey(1)
        self.front_camera =  (i3/255.0).reshape((1,IM_HEIGHT, IM_WIDTH, 3))

    def process_radar(self,data):
        radar_data = np.zeros((len(data), 4))
        for i, elt in enumerate(data):
            radar_data[i][0] = elt.altitude
            radar_data[i][1] = elt.azimuth 
            radar_data[i][2] = elt.depth / RADAR_RANGE
            radar_data[i][3] = elt.velocity / 30
        self.radar_data = radar_data.reshape((1,len(data), 4))

    def get_state(self):
        return np.array(self.front_camera,self.radar_data).reshape(2,)
    def step(self, action):
        if len(self.collision_hist) != 0:
            reward = -1.0
            done = True
         # elif car_world.vehicle.is_at_traffic_light():
        #     traffic_light_state = car_world.vehicle.get_traffic_light_state()
        #     if action == 0:  # Avancer
        #         if traffic_light_state == carla.TrafficLightState.Green:
        #             reward = 0.5
        #     elif action == 1 or action == 2:  # Tourner
        #         if traffic_light_state == carla.TrafficLightState.Green:
        #             reward = 0.5
        #     else :
        #             reward = -0.1
        else :
            v = self.vehicle.get_velocity()
            speed = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

            if speed > 10 :
                reward = 0.05

            elif speed > 20 :
                reward = 0.1

            elif speed > 30 :
                reward = 0.2

            elif speed > 40 :
                reward = 0.05
            elif speed > 50 :
                reward = -0.2
            else :
                reward = -0.02
            done = False
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0
        self.graph = tf.Graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        # Réseau pour les données de la caméra RGB
        input_rgb = Input(shape=(IM_HEIGHT,IM_WIDTH, 3))
        conv_rgb = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_rgb)
        flatten_rgb = Flatten()(conv_rgb)
        dense_rgb = Dense(32, activation='relu')(flatten_rgb)

        # Réseau pour les données du radar
        input_radar = tf.keras.Input(shape=(None, 4))

        # Réseau pour les données du radar
        lstm_radar = tf.keras.layers.LSTM(32)(input_radar)
        dense_radar = Dense(32, activation='relu')(lstm_radar)

        # Fusion des deux réseaux
        concatenated = Concatenate()([dense_rgb, dense_radar])
        dense_combined = Dense(64, activation='relu')(concatenated)

        # Couche de sortie pour les actions
        output = Dense(N_ACTIONS, activation='linear')(dense_combined)

        # Création du modèle final
        model = Model(inputs=[input_rgb, input_radar], outputs=output)
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = []
        for current_img,current_radar in zip((transition[0][0][0] for transition in minibatch ),(transition[0][1][0] for transition in minibatch )) :
            current_states.append((np.array(current_img).reshape((1,IM_HEIGHT,IM_WIDTH, 3)),np.array(current_radar).reshape((1,len(current_radar), 4))))
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states = []
        for new_current_img,new_current_radar in zip((transition[3][0][0] for transition in minibatch ),(transition[3][1][0] for transition in minibatch )) :
            new_current_states.append((np.array(new_current_img).reshape((1,IM_HEIGHT,IM_WIDTH, 3)),np.array(new_current_radar).reshape((1,len(new_current_radar), 4))))
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)


        with self.graph.as_default():
            future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        actions = [transition[1] for transition in minibatch]
        rewards = [transition[2] for transition in minibatch]
        dones = [transition[4] for transition in minibatch]
        for index in range(MINIBATCH_SIZE):
            if not dones[index]:
                max_future_q = np.max(future_qs_list[index])
                new_q = rewards[index] + DISCOUNT * max_future_q
            else:
                new_q = rewards[index]

            current_qs = current_qs_list[index]
            current_qs[actions[index]] = new_q

            X.append(current_states[index])
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        self.model.fit(np.array(X), np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)


        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        print(type(state))
        return self.model.predict(state)[0]

    def train_in_loop(self):
        X1 = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        X2 = np.random.uniform(size=(1, 200, 4)).astype(np.float32)
        y = np.random.uniform(size=(1, 4)).astype(np.float32)

        self.model.fit((X1,X2),y, verbose=False, batch_size=1)
        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)



class Agent:
# Créer un réseau de neurones avec TensorFlow
    def __init__(self):

        try :
            self.model = tf.keras.models.load_model("model.h5")
        except  OSError:
            self.model = self.create_q_model()

        self.target_model = self.create_q_model()
        self.target_model.set_weights(self.model.get_weights())



    def create_q_model(self): 
    # Réseau pour les données de la caméra RGB
        input_rgb = Input(shape=(IM_HEIGHT,IM_WIDTH, 3))
        conv_rgb = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_rgb)
        flatten_rgb = Flatten()(conv_rgb)
        dense_rgb = Dense(32, activation='relu')(flatten_rgb)

        # Réseau pour les données du radar
        input_radar = tf.keras.Input(shape=(None, 4))

        # Réseau pour les données du radar
        lstm_radar = tf.keras.layers.LSTM(32)(input_radar)
        dense_radar = Dense(32, activation='relu')(lstm_radar)

        # Fusion des deux réseaux
        concatenated = Concatenate()([dense_rgb, dense_radar])
        dense_combined = Dense(64, activation='relu')(concatenated)

        # Couche de sortie pour les actions
        output = Dense(N_ACTIONS, activation='linear')(dense_combined)

        # Création du modèle final
        model = Model(inputs=[input_rgb, input_radar], outputs=output)
        return model

    def choose_action(self, state, epsilon):
        if np.random.uniform() < epsilon:
            return np.random.randint(N_ACTIONS)
        else:
            q_values = self.model.predict(state, verbose = 0)
            return np.argmax(q_values[0])

    def train(self, memory,Running):
        while Running :
            if len(memory) < TRAIN_START:
                time.sleep(0.01)
                continue
            if len(memory) % 50 == 0 :
                minibatch = random.sample(memory,BATCH_SIZE)
                q_values = []
                for current_img,current_radar in zip((transition[0][0][0] for transition in minibatch) ,(transition[0][1][0] for transition in minibatch )) :
                    q_values.append(self.model.predict((np.array(current_img).reshape((1,IM_HEIGHT,IM_WIDTH, 3)),np.array(current_radar).reshape((1,len(current_radar), 4))),BATCH_SIZE,verbose = 0))
                    
                next_q_values = []
                for next_img,next_radar in zip([transition[3][0][0] for transition in minibatch ] ,[transition[3][1][0] for transition in minibatch ]) :
                    next_q_values.append(self.target_model.predict((np.array(next_img).reshape((1,IM_HEIGHT,IM_WIDTH, 3)),np.array(next_radar).reshape((1,len(next_radar), 4))),BATCH_SIZE,verbose = 0))

                actions = [transition[1] for transition in minibatch]
                rewards = [transition[2] for transition in minibatch]
                dones = [transition[4] for transition in minibatch]

            
                for i in range(BATCH_SIZE):
                    q_values[i][0][actions[i]] = rewards[i] + GAMMA * np.max(next_q_values[i]) * (1 - dones[i])

                states = []
                for current_img,current_radar in zip((transition[0][0][0] for transition in minibatch ),(transition[0][1][0] for transition in minibatch )) :
                    states.append((np.array(current_img).reshape((1,IM_HEIGHT,IM_WIDTH, 3)),np.array(current_radar).reshape((1,len(current_radar), 4))))
                for i in range(BATCH_SIZE):
                    LOSS  = self.model.fit(states[i], q_values[i],verbose= 0)
                self.target_model.set_weights(self.model.get_weights())

class ReplayMemory:
    def __init__(self):
        self.capacity = MEMORY_CAPACITY
        self.memory = []
    
    def push(self, transition):
        if len(self.memory) == self.capacity:
            self.memory.pop(0)
        self.memory.append(transition)
    
    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for sample in samples:
            state, action, reward, next_state, done = sample
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)



if __name__ == '__main__':
    FPS = 60
    # For stats
    ep_rewards = [-200]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    # Create agent and environment
    agent = DQNAgent()
    env = CarWorld()


    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    # Initialize predictions - forst prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    agent.get_qs((np.ones((1,IM_HEIGHT, IM_WIDTH, 3)),np.ones((1,200, 4))))

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        #try:

            env.collision_hist = []

            # Update tensorboard step every episode
            agent.tensorboard.step = episode

            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1

            # Reset environment and get initial state
            env.reset()
            current_state = (env.front_camera,env.radar_data)
            # Reset flag and start iterating until episode ends
            done = False
            episode_start = time.time()

            # Play for given number of seconds only
            while True:

                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Get action from Q table
                    print(env.radar_data)
                    print(env.front_camera)
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, 3)
                    # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                    time.sleep(1/FPS)

                new_state, reward, done, _ = env.step(action)

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                # Every step we update replay memory
                agent.update_replay_memory((current_state, action, reward, new_state, done))

                current_state = new_state
                step += 1

                if done:
                    break

            # End of episode - destroy agents
            for actor in env.actor_list:
                actor.destroy()

            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)


    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')



#############################################################################################################################################################################################################
    # # Créer le modèle principal et le modèle cible
    # car_world = CarWorld()
    # car_world.reset()
    # agent = Agent()
    

    # # Créer la mémoire de replay et l'optimiseur TensorFlow
    # mem = ReplayMemory()
    
    # # Boucle principale pour l'entraînement
    # episode_reward = 0.0
    # episode_steps = 0
    # episode_num = 0
    # epsilon = EPSILON_START
    # Running = True
    # state = None
    # # Entraîner le modèle avec un algorithme de Q-learning
    # trainer_thread = Thread(target=agent.train, daemon=True, args=(mem.memory,Running))
    # trainer_thread.start()
    # num = 0
    # while Running:
    #     time.sleep(1/FPS)
    #     if not state:
    #         image = car_world.image
    #         radar_data = car_world.radar_data
    #         state = (image,radar_data)
    #     # Choisir une action à effectuer
    #     action = agent.choose_action([image,radar_data], epsilon)
    #     # Effectuer l'action et obtenir la récompense
    #     if action == 0:  # Avancer
    #         car_world.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake = 0))
    #     elif action == 1:  # Tourner à gauche
    #         car_world.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=-0.5, brake = 0))
    #     elif action == 2:  # Tourner à droite
    #         car_world.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.5, brake = 0 ))
    #     elif action == 3: #frein
    #         car_world.vehicle.apply_control(carla.VehicleControl(steer=0,brake = 0.5 ))

    #     car_world.world.tick()
    #     image = car_world.image
    #     radar_data = car_world.radar_data
    #     next_state = (image, radar_data)
    #     collision = car_world.collisions_history
    #     if len(car_world.collisions_history) != 0:
    #         reward = -1.0
    #     # elif car_world.vehicle.is_at_traffic_light():
    #     #     traffic_light_state = car_world.vehicle.get_traffic_light_state()
    #     #     if action == 0:  # Avancer
    #     #         if traffic_light_state == carla.TrafficLightState.Green:
    #     #             reward = 0.5
    #     #     elif action == 1 or action == 2:  # Tourner
    #     #         if traffic_light_state == carla.TrafficLightState.Green:
    #     #             reward = 0.5
    #     #     else :
    #     #             reward = -0.1
    #     else:
    #         v = car_world.vehicle.get_velocity()
    #         speed = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

    #         if speed > 10 :
    #             reward = 0.05

    #         elif speed > 20 :
    #             reward = 0.1

    #         elif speed > 30 :
    #             reward = 0.2

    #         elif speed > 40 :
    #             reward = 0.05
    #         elif speed > 50 :
    #             reward = -0.2
    #         else :
    #             reward = -0.02
    #     episode_reward += reward
    #     episode_steps += 1
        
    #     # Ajouter l'expérience à la mémoire de replay
    #     transition = (state, action, reward, next_state, car_world.done)
    #     if(len(mem.memory) > MEMORY_CAPACITY) :
    #         mem.memory = mem.memory[len(mem.memory)//4:]
    #     mem.memory.append(transition)
    #     num += 1
    #     print(f"total steps: {num}  Speed: {speed}",end="\r")
    #     state = next_state
        
    #     # Mettre à jour le modèle cible
    #     if episode_steps % UPDATE_TARGET_FREQUENCY == 0:
    #         agent.target_model.set_weights(agent.model.get_weights())
        
    #     # Réduire la valeur d'epsilon
    #     if epsilon > EPSILON_END:
    #         epsilon -= EPSILON_DECAY
        
    #     # Afficher les statistiques de l'épisode
    #     if car_world.done:
    #         print(f'Episode {episode_num}: {episode_reward} (epsilon={epsilon:.5f}, loss={LOSS}, steps={episode_steps})')
    #         episode_reward = 0.0
    #         episode_steps = 0
    #         episode_num += 1
    #         for actor in car_world.actor_list:
    #             actor.destroy()

    #         if episode_num% 50 == 0:
    #             agent.model.save("model.h5")
    #         if episode_num > EPISODE_MAX :
    #             agent.model.save("model.h5")
    #             Running = False
    #             break
    #         else :
    #             car_world.reset()
    #             state = None
    #             time.sleep(0.5)
