import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import numpy as np
from collections import namedtuple, deque

Step = namedtuple('Step', ['state', 'action', 'reward', 'done', 'lstm'])


class NStepProgress:
    
    def __init__(self, env, ai, n_step):
        self.ai = ai 
        self.rewards = []
        self.env = env 
        self.n_step = n_step 

    def __iter__(self): #funcion para jugar el juego y devolver las experiencias generadas por el agente.
        state = self.env.reset() #reseteando el juego.
        history = deque() 
        reward = 0.0 
        is_done = True
        end_buffer = []

        while True:
            if is_done:
                cx = Variable(torch.zeros(1,256))
                hx = Variable(torch.zeros(1,256))
            else:
                cx = Variable(cx.data)
                hx = Variable(hx.data)

            action, (hx, cx) = self.ai(Variable(torch.from_numpy(np.array([state], dtype = np.float32))), (hx, cx)) #calculando la accion del siguiente estado
            end_buffer.append((state, action))
            
            while len(end_buffer) > 3: 
                del end_buffer[0]

            t = action[0][0]

            if(t == 1):     #left
                print("left")
            elif (t == 2):  #right
                print("right")
            elif (t == 3):  #roll
                print("jump")
            elif (t == 4):  #jump
                print("roll")
            elif (t == 0):  #no op
                print("do nothing")

        
            next_state, r, is_done, _ = self.env.step(action) 

            if(is_done):
                print("\nGame Ended\n")
                if len(end_buffer)>=3:
                    state, action = end_buffer[-3]
                    history.pop() 
                r=-10
            reward += r
            history.append(Step(state = state, action = action, reward = r, done = is_done, lstm = (hx, cx))) 
            
        
            while len(history) > self.n_step + 1:
                history.popleft()
            if len(history) == self.n_step + 1:
                yield tuple(history)
            state = next_state
          
            if is_done:
                if len(history) > self.n_step + 1:
                    history.popleft()
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()
         
                self.rewards.append(reward)
                reward = 0.0
                state = self.env.reset()
                end_buffer=[]
                history.clear()
    
    def rewards_steps(self): 
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps