import numpy as np
                        
input_dataset = np.array(([5,5],[2,1],[6,6],[3,5],[1,1],[2,1],[3,3],[3,4],[5,5]), dtype = float) #input values of the dataset
output_dataset = np.array(([1],  [0],  [1],  [0],  [1],  [0],  [1],  [0]), dtype = float) # output values of the dataset: near 1 = squarre, near 0 = rectangle

input_dataset = input_dataset/np.amax(input_dataset, axis=0) #put the values of the neurons between 0 an 1 for an easier readability

wanted_values = np.split(input_dataset,[8])[0] #split allow us to get only the wanted values, here : the 8 first
predict = np.split(input_dataset,[8])[1]

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

    def backward(self, wanted_values, output_dataset, output_values): #retro-propagation function
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

    def training(self, wanted_values, output_dataset): #training function of the AI
        output_values = self.forward(wanted_values)
        self.backward(wanted_values,output_dataset,output_values)

    def prediction(self): #print the prediciton of the AI after training
        print("donnée prédite après entrainement: ")
        print("Entrée : \n" + str(predict))
        print("Sortie : \n" + str(self.forward(predict)))

        if(self.forward(predict) < 0.5):
            print("L'IA a trouvé un rectangle \n")
        else:
            print("L'IA a trouvé un carré \n")


brain = Neural_Network()

for i in range(50000): #training loop, we put in parameter number of training the number of training that we want our AI to do
    print("entrainement n° " + str(i) + "\n")
    print("Sortie du dataset: \n" + str(output_dataset))
    print("\nSortie trouver par l'IA: \n" + str(np.matrix.round(brain.forward(wanted_values),2)))
    print("\n")
    brain.training(wanted_values,output_dataset)

brain.prediction()
