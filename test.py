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
from threading import Thread, Lock
import time
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
EPSILON_DECAY = 0.9995
MEMORY_CAPACITY = 2000
TRAIN_START = 1000
TRAIN_INTERVAL = 50
TARGET_UPDATE_INTERVAL = 100
EPISODE_MAX = 500
RADAR_RANGE = 100
FPS = 30
LOSS = None
UPDATE_TARGET_FREQUENCY = 1000

# Valeur minimale d'epsilon
EPSILON_END = 0.05

key = Lock()
class CarWorld:

    # Se connecter au serveur CARLA et créer un monde
    def __init__(self):
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()


    def reset(self) :
        self.collisions_history = []        
        self.actor_list = []
        self.done = False


        # Créer un véhicule avec des capteurs de caméra et de radar
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
        self.collision_sensor.listen(lambda event: self.process_collision(event))
        self.actor_list.append(self.collision_sensor)


    def process_img(self,image):
        i = np.array(image.raw_data)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        cv2.imshow("", i3)
        cv2.waitKey(1)
        self.image =  i3.reshape((1,IM_HEIGHT, IM_WIDTH, 3))

    def process_radar(self,data):
        radar_data = np.zeros((len(data), 4))
        for i, elt in enumerate(data):
            radar_data[i][0] = elt.altitude
            radar_data[i][1] = elt.azimuth 
            radar_data[i][2] = elt.depth / RADAR_RANGE
            radar_data[i][3] = elt.velocity / 30
        self.radar_data = radar_data.reshape((1,len(data), 4))

    def process_collision(self, event):
        self.collisions_history.append(event)
        self.done = True


class Agent:
    def __init__(self):

        try :
            self.model = tf.keras.models.load_model("model.h5")
        except  OSError:
            self.model = self.create_q_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer, loss=tf.keras.losses.BinaryCrossentropy())

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
            key.acquire()
            q_values = self.model.predict(state, verbose = 0)
            key.release()
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
                    self.model.fit(states[i], q_values[i],verbose= 0)

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
    # Créer le modèle principal et le modèle cible
    car_world = CarWorld()
    car_world.reset()
    agent = Agent()
    

    # Créer la mémoire de replay
    mem = ReplayMemory()
    
    # Boucle principale pour l'entraînement
    episode_reward = 0.0
    episode_steps = 0
    episode_num = 0
    epsilon = EPSILON_START
    Running = True
    state = None
    # Entraîner le modèle avec un algorithme de Q-learning
    trainer_thread = Thread(target=agent.train, daemon=True, args=(mem.memory,Running))
    trainer_thread.start()
    num = 0
    while Running:
        time.sleep(1/FPS)
        if not state:
            image = car_world.image
            radar_data = car_world.radar_data
            state = (image,radar_data)
        # Choisir une action à effectuer
        action = agent.choose_action([image,radar_data], epsilon)

        # Effectuer l'action et obtenir la récompense
        if action == 0:  # Avancer
            car_world.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake = 0))
        elif action == 1:  # Tourner à gauche
            car_world.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=-0.5, brake = 0))
        elif action == 2:  # Tourner à droite
            car_world.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.5, brake = 0 ))
        elif action == 3: # Freiner
            car_world.vehicle.apply_control(carla.VehicleControl(steer=0,brake = 0.5 ))

        car_world.world.tick()
        image = car_world.image
        radar_data = car_world.radar_data
        next_state = (image, radar_data)
        collision = car_world.collisions_history
        if len(car_world.collisions_history) != 0:
            reward = -1.0
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
        else:
            v = car_world.vehicle.get_velocity()
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
        episode_reward += reward
        episode_steps += 1
        
        # Ajouter l'expérience à la mémoire de replay
        transition = (state, action, reward, next_state, car_world.done)
        if(len(mem.memory) > MEMORY_CAPACITY) :
            mem.memory = mem.memory[len(mem.memory)//4:]
        mem.memory.append(transition)
        num += 1
        print(f"total steps: {num}  Speed: {speed}",end="\r")
        state = next_state
        
        # Mettre à jour le modèle cible
        if num % UPDATE_TARGET_FREQUENCY == 0:
            agent.target_model.set_weights(agent.model.get_weights())
        
        # Réduire la valeur d'epsilon
        if epsilon > EPSILON_END:
            epsilon = max(EPSILON_FINAL,epsilon * EPSILON_DECAY)
        
        # Afficher les statistiques de l'épisode
        if car_world.done:
            print(f'Episode {episode_num}: {episode_reward} (epsilon={epsilon:.5f}, steps={episode_steps})')
            episode_reward = 0.0
            episode_steps = 0
            episode_num += 1
            for actor in car_world.actor_list:
                actor.destroy()

            if episode_num% 50 == 0:
                agent.model.save("model.h5")
            if episode_num > EPISODE_MAX :
                agent.model.save("model.h5")
                Running = False
                break
            else :
                car_world.reset()
                state = None
                time.sleep(0.5)
