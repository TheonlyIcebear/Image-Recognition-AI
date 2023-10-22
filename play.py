from multiprocessing import Process, Queue
from utils.model import Model
from utils.game import Game
from numba import jit, cuda
from PIL import Image
from tqdm import tqdm
import threading, requests, hashlib, random, numpy as np, math, copy, time, json, os

class Main:
    def __init__(self):
        file = json.load(open('model-training-data.py', 'r+'))

        self.brain = [self.pad(file[0])] + [np.array(file[1])] + file[2:]
        self.play()

    def pad(self, model):
        model = model[:]

        for layer_idx, layer in enumerate(model):
            for node_idx, node in enumerate(layer):
                max_weight = max(len(layer[0]) for layer in model)
                weights = len(node)
                node = node + np.zeros((max_weight - weights)).tolist()

                model[layer_idx][node_idx] = node
            
            max_height = max(len(layer) for layer in model)
            height = len(layer)
            layer = layer + np.zeros((max_height - height, max_height)).tolist()

            model[layer_idx] = layer

        return np.array(model)

    def play(self):
        model, heights, hidden_activation_function, output_activation_function, cost_function = self.brain
        model = Model(
            model=model,
            heights=heights,
            hidden_function = hidden_activation_function,
            output_function = output_activation_function,
            cost_function = cost_function
        )

        options = ['airplane', 'bicycle', 'boat', 'motorbus', 'motorcycle', 'seaplane', 'train', 'truck', 'Custom File']

        while True:
            for idx, option in enumerate(options):
                print(f"{idx + 1}. {option.capitalize()}")

            print('\n')

            print("Enter a integer 1-9")
            choice = int(input(">> "))
            
            if choice == 9:
                print("Enter Image Url: ")
                url = input(">> ")
                image = Image.open(requests.get(url, stream=True).raw)
            else:

                folder = options[choice - 1]
                
                filename = random.choice(os.listdir(folder))
                image = Image.open(f'{folder}\\{filename}')

            image.resize((128, 128)).convert('L').show()

            normalized_image = np.array(image
                    .resize((128, 128))
                    .convert('L')
                ).flatten() / 255

            print(normalized_image)

            model_outputs = model.eval(
                input = normalized_image
            )[-1]

            answer = np.argmax(model_outputs)
            print(f"It's a {options[answer]}")
            print(model_outputs, '\n\n')



if __name__ == '__main__':
    Main()