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


    

class Neural_Network(object): #creation of the neural network class
    def __init__(self):
        self.inputnum = 2 #define the number of neurons in the input layer
        self.outputnum = 1 #define the number of neurons in the output layer
        self.hiddennum = 3 #define the number of neurons in the hidden layer

        #randomlly create the synaps weights
        self.W1 = np.random.randn(self.inputnum, self.hiddennum) #matrix 2*3
        self.W2 = np.random.randn(self.hiddennum, self.outputnum) #matrix 3*1

    def forward(self, wanted_values): #forward propagation funtion
        self.input_matrix_product = np.dot(wanted_values,self.W1) #matrix multiplication between our input values and the synaps weight 
        self.hidden_values = self.sigmoid(self.input_matrix_product) #we apply the sigmoid function to the matrix product, we obtain the hidden layer values
        self.hidden_matrix_product = np.dot(self.hidden_values,self.W2) #matrix multiplication between the hidden values and the synaps weight
        output_values = self.sigmoid(self.hidden_matrix_product) #we apply the sigmoid function to the matrix product, we obtain the output layer values
        return output_values
    
    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s)) #this is the sigmoid function

    def sigmoid_prime(self, s): #compute the derivative of the sigmoid function
        return s * (1 - s)

    def backward(self, wanted_values, output_values): #retro-propagation function
        global output_dataset

        #output neurons layer
        self.output_error = output_dataset - output_values #compute the margin of error
        self.output_delta = self.output_error * self.sigmoid_prime(output_values) #compute the delta of the margin of error

        #hidden neurons layer
        self.hidden_values_error = self.output_delta.dot(self.W2.T) #compute the margin of error
        self.hidden_values_delta = self.hidden_values_error * self.sigmoid_prime(self.hidden_values) #compute the delta of the margin of error

        #update the synaps weight
        self.W1 += wanted_values.T.dot(self.hidden_values_delta) #we add to the synaps weight between the input layer and the hidden layer 
                                                                 #the matrix product of the input values with the delta ou le margin of error of the hidden neurons layer

        self.W2 += self.hidden_values.T.dot(self.output_delta) #we add to the synaps weight between the hidden layer and the output layer
                                                          #the matrix product of the hidden values with the delta ou le margin of error of the output neurons layer

    def training(self, wanted_values): #training function of the AI
        global output_dataset

        output_values = self.forward(wanted_values)
        self.backward(wanted_values,output_values)

    def prediction(self): #print the prediciton of the AI after training
        predict = get_predict ()
        print("donnée prédite après entrainement: ")
        print("Entrée : \n" + str(predict))
        print("Sortie : \n" + str(self.forward(predict)))

        if(self.forward(predict) < 0.5):
            print("L'IA a trouvé un rectangle \n")
        else:
            print("L'IA a trouvé un carré \n")


filling_dataset()
wanted_values = get_wanted_values()

brain = Neural_Network()

for i in range(30000): #training loop, we put in parameter number of training the number of training that we want our AI to do
    print("entrainement n° " + str(i) + "\n")
    print("Sortie du dataset: \n" + str(output_dataset))
    print("\nSortie trouver par l'IA: \n" + str(np.matrix.round(brain.forward(wanted_values),2)))
    print("\n")
    brain.training(wanted_values)

brain.prediction()
