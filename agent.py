from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Activation,Dropout,Flatten
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import time
class Agent ():
    def __init__(self,OBSERVATION_SPACE_VALUES,ACTION_SPACE_SIZE):
        self.ACTION_SPACE_SIZE = ACTION_SPACE_SIZE
        self.OBSERVATION_SPACE_VALUES = OBSERVATION_SPACE_VALUES

        self.REPLAY_MEMORY_SIZE = 50000
        self.MIN_MEMORY_SIZE = 1000
        self.MINIBARCH_SIZE = 64
        self.DISCOUNT = 0.99
        self.UPDATE_TARGET_EVERY = 5

        # fit
        self.model = self.create_model()
        #predict
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque (maxlen = self.REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0
    def save_model(self):
        print ('saveing model')
        self.target_model.save (str(time.time())+'.h5')

    def create_model(self):
        '''ritorna un modello keras per l agente'''
        model = Sequential ()
        model.add (Dense (256,input_shape = self.OBSERVATION_SPACE_VALUES))
        model.add (Dense(160))
        model.add (Dense (self.ACTION_SPACE_SIZE,activation = 'linear'))
        model.compile(loss='mse',optimizer = Adam(lr=0.001),metrics = ['accuracy'])
        return model
    def update_replay_memory (self,transition):
        self.replay_memory.append (transition)
    def get_qs(self,state):
        return self.model.predict(np.array(state).reshape (-1,*state.shape)/255)[0]
    def train (self,terminal_state,step):
        if len (self.replay_memory) < self.MIN_MEMORY_SIZE:
            return
        minibatch = random.sample (self.replay_memory,self.MINIBARCH_SIZE)
        current_states = np.array ([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array ([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index,(current_state,action,reward,new_current_state,done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else :
                new_q = reward
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            X.append (current_state)
            y.append (current_qs)
        self.model.fit(np.array(X)/255,np.array(y),batch_size = self.MINIBARCH_SIZE,verbose = 0,shuffle = False )if terminal_state else None
        if terminal_state:
            self.target_update_counter +=1
        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())