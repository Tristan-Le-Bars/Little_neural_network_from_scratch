import numpy as np

example_num = int(input("how much example do you want to give to the AI ?\n"))
input_dataset = np.zeros((example_num + 1, 2), dtype = float)
output_dataset = np.zeros((example_num, 1), dtype = float)


def filling_dataset ():
    global input_dataset
    global output_dataset
    global example_num
    i = 0
    param1 = 0
    param2 = 0
    obj_type = -1
    
    while i < int(example_num):
        param1 = input("Parameter 1 of the object n°" + str(i) + ": ")
        param2 = input("Parameter 2 of the object n°" + str(i) + ": ")
        obj_type = input("type of the object (0 or 1) :")
        input_dataset[i,:] = [int(param1), int(param2)]
        output_dataset[i,:] = [int(obj_type)]
        i = i + 1
    param1 = input("Parameter 1 of the object that the AI have to find the type : ")
    param2 = input("Parameter 2 of the object that the AI have to find the type : ")
    input_dataset[i,:] = [int(param1), int(param2)]
    input_dataset = input_dataset/np.amax(input_dataset, axis = 0)


def get_wanted_values ():
    global input_dataset
    global example_num

    wanted_values = np.split(input_dataset, [example_num])[0]
    return wanted_values

def get_predict ():
    global input_dataset
    global example_num

    predict = np.split(input_dataset, [example_num])[1]
    return predict   

class Neural_Network(object):
    def __init__(self):
        self.inputnum = 2
        self.outputnum = 1
        self.hiddennum = 3

        self.W1 = np.random.randn(self.inputnum, self.hiddennum)
        self.W2 = np.random.randn(self.hiddennum, self.outputnum)

    def forward(self, wanted_values):
        self.input_matrix_product = np.dot(wanted_values,self.W1) 
        self.hidden_values = self.sigmoid(self.input_matrix_product)
        self.hidden_matrix_product = np.dot(self.hidden_values,self.W2)
        output_values = self.sigmoid(self.hidden_matrix_product)
        return output_values
    
    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def sigmoid_prime(self, s):
        return s * (1 - s)

    def backward(self, wanted_values, output_values):
        global output_dataset
        
        #output neurons layer
        self.output_error = output_dataset - output_values
        self.output_delta = self.output_error * self.sigmoid_prime(output_values)
        #hidden neurons layer
        self.hidden_values_error = self.output_delta.dot(self.W2.T)
        self.hidden_values_delta = self.hidden_values_error * self.sigmoid_prime(self.hidden_values)
        #update the synaps weight
        self.W1 += wanted_values.T.dot(self.hidden_values_delta) 
        self.W2 += self.hidden_values.T.dot(self.output_delta) 
        
    def training(self, wanted_values):
        global output_dataset

        output_values = self.forward(wanted_values)
        self.backward(wanted_values,output_values)

    def prediction(self):
        predict = get_predict ()
        print("Prediciton after training: ")
        print("Output : \n" + str(self.forward(predict)))

        if(self.forward(predict) < 0.5):
            print("The AI have found that the type of the object is 0 \n")
        else:
            print("The AI have found that the type of the object is 1")


filling_dataset()
wanted_values = get_wanted_values()

brain = Neural_Network()

for i in range(30000):
    print("Training number " + str(i) + "\n")
    print("Dataset output: \n" + str(output_dataset))
    print("\nOutput found by the AI: \n" + str(np.matrix.round(brain.forward(wanted_values),2)))
    print("\n")
    brain.training(wanted_values)

brain.prediction()
