from multiprocessing import Process, Queue
from utils.model import Model
from numba import jit, cuda
from PIL import Image
from tqdm import tqdm
import multiprocessing, threading, hashlib, random, numpy as np, copy, gzip, math, time, json, os

class Main:
    def __init__(self, tests_amount, generation_limit, learning_rate, momentum_conservation, weight_decay, cost_limit, dimensions, threads):
        self.tests = tests_amount
        self.trials = generation_limit
        self.learning_rate = learning_rate
        self.momentum = momentum_conservation
        self.wd = weight_decay
        self.layers, self.height = dimensions[0]
        self.shape = dimensions[1]
        self.threads = threads
        self.cost_limit = cost_limit
        
        self.save_queue = Queue()
        self.queue = Queue()
        self.children = 0
        self.generations = 0
        self.average_cost = 0
        self.model = []
        self.layers += 1
        self.inputs = 128 ** 2
        
        self.options = ['airplane', 'bicycle', 'boat', 'motorbus', 'motorcycle', 'seaplane', 'train', 'truck']
        
        try:
            with gzip.open('model-training-data.gz', 'rb') as file:
                file = json.loads(file.read().decode())

            network = [self.pad(file[0])] + [np.array(file[1])] + file[2:]
        except Exception as e:
            print(e)
            network = self.build()
            open('model-training-data.gz', 'w+').close()
            self.save_queue.put(network)

        print(network[1:])

        threading.Thread(target=self.manager, args=(network,)).start()

        self.save()

    def depad(self, model):
        model = model.tolist()
        max_height = max(len(layer) for layer in model)

        for layer_idx, layer in enumerate(model):
            for node_idx, node in enumerate(layer):
                if 0 in node:

                    end = node.index(0)
                    node = node[:end]

                    model[layer_idx][node_idx] = node
            
            filler = np.zeros(max_height).tolist()
            if filler in layer:

                end = layer.index(filler)
                layer = layer + np.zeros((max_height - height, max_height))

                model[layer_idx] = layer

        return model

    def pad(self, model):
        model = model[:]

        for layer_idx, layer in tqdm(enumerate(model)):
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

    def save(self):
        queue = self.save_queue

        while True:
            network = queue.get()

            try:
                data = json.dumps([self.depad(network[0]), network[1].tolist()] + network[2:]).encode()
            except Exception as e:
                print(e)
                data = None

            if data:
                with gzip.open('model-training-data.temp', 'wb') as file:
                    file.write(data)
                    file.close()

                os.remove('model-training-data.gz')
                os.rename('model-training-data.temp', 'model-training-data.gz')
            

    # Manages threads
    def manager(self, network=None):
        threads = []
        average_accuracy = np.array([])

        model = network[0]
        momentum = np.zeros(model.shape)

        queue = self.queue
        
        for count in range(self.threads):
            receive_queue = Queue()

            thread = multiprocessing.Process(target=self.worker, args=(receive_queue, count,))
            threads.append([thread, receive_queue])

        for thread in threads:
            thread[0].start()


        cost = 0
        old_accuracy = -1
        self.average_cost = None

        cost_overtime = []

        for _ in range(self.trials):

            network = [model] + network[1:]

            self.save_queue.put(network)
            
            count += 1
            self.generations += 1 
            self.model = model

            self.children = self.threads

            for thread in threads:
                thread[1].put(network)

            gradient = np.zeros(model.shape)

            for _ in tqdm(range(self.threads)):
                _cost, gradient_map = queue.get()
                gradient += gradient_map
                
                cost += _cost
                self.children -= 1

            self.average_cost = cost / (self.threads * self.tests)

            os.system('cls')

            print(f"Average Cost: {self.average_cost}, Generations: {self.generations}, Live: {self.children}")

            cost_overtime.append(self.average_cost)

            json.dump(cost_overtime, open('generations.json', 'w+'), indent=2)

            if self.average_cost < self.cost_limit:
                print(f"Cost Minimum Reached: {self.cost_limit}")
                os.system("PAUSE")

            
            
            momentum = (gradient * self.learning_rate ) + (momentum * self.momentum)

            model -= momentum / self.tests
            
            cost = 0

    def worker(self, receieve_queue=None, thread_index=0):
        generations = 0
        options = self.options
        start = 0

        while True:
            generations += 1
            model, heights, hidden_activation_function, output_activation_function, cost_function = receieve_queue.get()
            model = Model(
                model=model,
                heights=heights,
                hidden_function=hidden_activation_function, 
                output_function=output_activation_function, 
                cost_function=cost_function
            )

            gradient = np.zeros(model.model.shape)
            trials = self.tests
            cost = 0

            load_amount = self.tests // self.threads

            for count in range(self.tests):

                choice = random.randint(0, 7)
                
                folder = options[choice]

                filename = random.choice(os.listdir(folder))
                
                image = Image.open(f'{folder}\\{filename}')

                normalized_image = np.array(image
                        .resize((128, 128))
                        .convert('L')
                    ).flatten() / 255

                model_outputs = model.eval(
                    input=normalized_image, 
                    dropout_rate=0,
                    training=True
                )

                expected_output = np.zeros(8)
                expected_output[choice] = 1

                _gradient, _cost = model.gradient(
                        model_outputs, 
                        expected_output
                    )

                gradient += _gradient

                # cost += 1 * (np.argmax(model_outputs[-1]) == choice)
                cost += _cost

            start += self.tests * self.threads

            self.queue.put([cost, gradient])

            del model, heights, hidden_activation_function, output_activation_function, cost_function
                    
    def build(self):
        inputs = self.inputs
        height = self.height
        shape = self.shape

        heights = np.full(self.layers, self.height)

        model = [*range(self.layers)]

        if shape:
            heights[[*shape.keys()]] = [*shape.values()]
            
        heights = np.append([inputs], heights)
        heights[-1] = 8

        variance = 0.15
        
        for idx, inputs in enumerate(heights[:-1]):
            model[idx] = np.random.uniform(-variance, variance, (height, inputs + 1)).tolist()


        return [self.pad(model), heights, "relu", "softmax", "cross_entropy"]

if __name__ == "__main__":
    Main(
        tests_amount = 32, # The length of the tests,
        generation_limit = 1000000, # The amount of generations the model will be trained through
        learning_rate = 0.05, # How fast the model learns, if too low the model will train very slow and if too high it won't train
        momentum_conservation = 0.9, # What percent of the previous changes that are added to each weight in our gradient descent
        weight_decay = 0.01,
        cost_limit = 0.0,
        dimensions = [
            [2, 156], 
            {
                1: 96,
                2: 64
            }
        ],  # The length and height of the model
        threads = 4  # How many concurrent threads to be used
    )