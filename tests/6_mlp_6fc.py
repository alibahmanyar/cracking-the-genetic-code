import sys
sys.path.append('../')

from tokenizerOHE import letter_to_index, letter_to_tensor, sequence_to_tensor, loadData, codon_to_tensor, \
    tensor_to_codon, index_to_tensor, updateANDsave_heatmap
from tokenizerOHE import nucleotideLetters, aminoAcidLetters, nucleotideLetterList, \
    aminoAcidOneLetterList, codonsList, aaWeight

import random
import torch
import torch.utils.data as data
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence # for sequences padding to get the same length for batch size >1
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import math
import sys

FIG_LABEL = "MLP with 6 hidden layers"
FIGS_PATH = "./mlp-6fc/"

from pathlib import Path
Path(FIGS_PATH).mkdir(parents=True, exist_ok=True)

filePATH = r"../DATA/"
fileOne = r"humanORFs.txt"
fileTwo = r"humanProteins.txt"
cdsIDmRNA, ORFs, cdsIDprot, proteins = loadData(filePATH+fileOne, filePATH+fileTwo)

# Data Loading: get amino acid labels (ground truth) associated to mRNA sequences:
# Create the target (labels) file for ORFs sequences:
# In the 'humanORFs.txt' file, there are 69,768 ORF sequences of nucleotides of variable lengths.
# In the 'humanProteins.txt' file, there are 69,768 amino acid sequences of the same variable length as before but divided by 3.
# The ORFs file will be converted to a set of 69,768 tensors with dimension [ORFseq_length, 1, 4].
# The proteins file will be converted to a set of 69,768 tensors with dimension[ORFseq_length/3, 1, 21].
# The mapping of the first file to the second implicitly assigns 21 unique target values to all encountered nucleotide triplets.
# The outcome is a multinomial outcome: given a triplet of nucleotides, you implicitly assign a target value taken from a set
# of 21 amino acid residues (STOP * included).

root_dir = r"../DATA/"
target_file = r"targets.txt"
created_target_file = open(root_dir + target_file, 'w')

for orf_idx in range(len(cdsIDmRNA)):
    line = format(orf_idx, '5d') + '\t' + str(proteins[orf_idx]) + '\n'
    created_target_file.write(line)
created_target_file.close()

import pandas as pd

from torch.utils.data import Dataset

class sequencesDataset(data.Dataset):
    def __init__(self, root_dir, target_file, train=True, transform=None):
        """Initializes a dataset containing ORFs and proteins sequences."""
        super(sequencesDataset, self).__init__()
        # your code
        # df = pd.read_table(root_dir+target_file)
        self.root_dir = root_dir
        # self.target_file = target_file
        self.labels = pd.read_table(root_dir + target_file, header=None)
        #self.labels = proteins
        self.train = train
        self.transform = transform

    def __len__(self):
        """returns size of datatset."""
        return len(self.labels)

    def __getitem__(self, index):
        """Returns the index-th data item of the dataset."""
        ''' and returns the target label.'''
        # your code
        # super().__getitem__(index)
        #img = Image.open(root_dir + str(index) + '.jpg')
        ORF_seq = sequence_to_tensor(ORFs[index], nucleotideLetters)
        #if self.transform:
        #    ORF_seq = self.transform(ORF_seq)
        #y_label = torch.tensor(int(self.labels.iloc[index, 1]))
        y_label = sequence_to_tensor(proteins[index], aminoAcidLetters)
        return ORF_seq, y_label

# Let's have a quick look to some of the samples:
my_dataset = sequencesDataset(root_dir, target_file, train=True, transform=None)
# have a look at the structure of 'my_dataset':
# print the labels:
print('labels')
print(my_dataset.labels[0:10])
# print the length of the ORF and Proteins dataset:
print('length of my_dataset = ', my_dataset.__len__())
# return the ORF and its label for a given index:
idx = 12500
orf, targetprotein = my_dataset.__getitem__(idx)
print('index = ', idx, ', the orf for this index = \n', ORFs[idx])
print('index = ', idx, ', the protein for this index = \n', proteins[idx])
print('the length of this protein is:', len(proteins[idx]))
#print('index = ', idx, ', the target protein for this index = \n', targetprotein.item())
print('index = ', idx, ', the target protein for this index = \n', targetprotein.shape)

# Let's create 'myCodonSet', a dataset associating a list of codons to the target amino acid they map to.
codonsNumber = 640000 # will be the row number of 'myCodonSet' (640,000).
frameshift = int(3)
class codonsDataset(data.Dataset):
    def __init__(self, root_dir, target_file, train=True, transform=None):
        """Initializes a dataset containing ORFs and proteins sequences."""
        super(codonsDataset, self).__init__()
        # your code
        # df = pd.read_table(root_dir+target_file)
        self.root_dir = root_dir
        # self.target_file = target_file
        self.labels = pd.read_table(root_dir + target_file, header=None)
        #self.labels = proteins
        self.train = train
        self.transform = transform
        # stack 640,000 codons sequentially taken from randomly picked orfs and associated proteins:
        codons_List = list()
        target_aa_List = list()
        count_codons = 1
        while count_codons < codonsNumber:
            orf_picked, protein_picked, _, _ = random_training_example(ORFs, proteins)
            for frame in range(int(len(protein_picked))):
                codons_List.append(orf_picked[frame*frameshift:(frame+1)*frameshift])
                target_aa_List.append(protein_picked[frame])
                count_codons += 1
                if count_codons > codonsNumber:
                    break
        print('count_codons=', count_codons)
        self.codons = codons_List
        self.targets = target_aa_List

    def __len__(self):
        """returns size of codons datatset."""
        return len(self.codons)

    def __getitem__(self, index):
        '''Returns the index-th data item of the dataset.'''
        ''' and returns the target aa label.'''
        # your code
        # super().__getitem__(index)
        codon_tensor = codon_to_tensor(self.codons[index], codonsList)
        target_tensor = letter_to_tensor(self.targets[index], aminoAcidLetters)
        return codon_tensor, target_tensor

# This function provides the alphabet sequence from the tensor with 4 dimensions:
def tensor_to_sequence(t, alphabet):
    # this function returns the sequence from the tensor, taking into account possible padding zeros in the
    # tensor along the sequence_length axis (which is normally axis 1, as axis 0 is the batch size).
    cumulated_string = ''
    for i in range(t.shape[1]):
        if len(alphabet) == 21:
            # try (if there happens to be at least a 1.0, it will find an index, else: there are only padding zeros
            # when there are padding zeros, you can stop the sequence because you are over the <eos>.
            try:
                cumulated_string += aminoAcidLetters[t[0, i, 0, :].tolist().index(1.0)]
            except ValueError: # this exception is raised when padding zeros are encountered and the index method raised a ValueError:
                break
        if len(alphabet) == 4:
            # try (if there happens to be at least a 1.0, it will find an index, else: there are only padding zeros
            # when there are padding zeros, you can stop the sequence because you are over the <eos>.
            try:
                cumulated_string += nucleotideLetters[t[0, i, 0, :].tolist().index(1.0)]
            except ValueError: # this exception is raised when padding zeros are encountered and the index method raised a ValueError:
                break
    return cumulated_string

# This function provides the alphabet sequence from the tensor with 3 dimensions:
def tensor3_to_sequence(t3, alphabet):
    cumulated_string = ''
    for i in range(t3.shape[0]):
        if len(alphabet) == 21:
            # try (if there happens to be at least a 1.0, it will find an index, else: there are only padding zeros
            # when there are padding zeros, you can stop the sequence because you are over the <eos>.
            try:
                cumulated_string += aminoAcidLetters[t3[i, 0, :].tolist().index(1.0)]
            except ValueError: # this exception is raised when padding zeros are encountered and the index method raised a ValueError:
                break
        if len(alphabet) == 4:
            # try (if there happens to be at least a 1.0, it will find an index, else: there are only padding zeros
            # when there are padding zeros, you can stop the sequence because you are over the <eos>.
            try:
                cumulated_string += nucleotideLetters[t3[i, 0, :].tolist().index(1.0)]
            except ValueError:
                break
    return cumulated_string

def random_training_example(orf_sequences, protein_sequences):
    # this function returns a randomly drawn orf and its associated target protein along with their tensors
    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx], random_idx

    orf_seq, picked_idx = random_choice(orf_sequences)
    target_protein = protein_sequences[picked_idx]
    orf_tensor = sequence_to_tensor(orf_seq, nucleotideLetters)
    protein_tensor = sequence_to_tensor(target_protein, aminoAcidLetters)
    return orf_seq, target_protein, orf_tensor, protein_tensor

# Let's have a quick look to some of the samples:
myCodonSet = codonsDataset(root_dir, target_file, train=True, transform=None)
# have a look at the attributes and methods of 'myCodonSet':
# print the aa:
#print('aa targets')
#print(myCodonSet.targets[0:10])
# print the length of the ORF and Proteins dataset:
#print('length of myCodonSet = ', myCodonSet.__len__())
# return the codon and its aa for a given index:
#idx = 640000-1
#cod, targetAA= myCodonSet.__getitem__(idx)
#print('index = ', idx, ', the codon for this index = \n', cod)
#print('index = ', idx, ', the protein for this index = \n', targetAA)
#print('The codon is ', codonsList[cod[0].tolist().index(1.0)])
#print('The codon is ', tensor_to_codon(cod, codonsList))
#print('The aa is', tensor3_to_sequence(targetAA.unsqueeze(0), aminoAcidLetters))

# build the training set and test set for the codons:
#----------------------------------------------------
train_codon_set, test_codon_set = torch.utils.data.random_split(myCodonSet, [myCodonSet.__len__()-64000, 64000])
from torch.utils.data import DataLoader
batch_size_of_codons = 64
train_codon_loader = DataLoader(train_codon_set, batch_size=batch_size_of_codons, shuffle=True, num_workers=0)
test_codon_loader = DataLoader(test_codon_set, batch_size=batch_size_of_codons, shuffle=False, num_workers=0)


# Let's have a look on the data in the batches...
#print('size of train codon set=', len(train_codon_set))
#print('size of test codon set=', len(test_codon_set))
#print('train_loader')
#next_batch_codons, next_batch_aas = next(iter(train_codon_loader))
#print('next batch of codons tensor', next_batch_codons.size())
#print('next batch of codons tensor', next_batch_codons)
#print('last codon in batch tensor[-1].tolist()', next_batch_codons[-1].tolist())
#print('next codon train=', tensor_to_codon(next_batch_codons[-1], codonsList))
#print('next aa train=', tensor3_to_sequence(next_batch_aas[-1].unsqueeze(0), aminoAcidLetters))

#next_batch_codons, next_batch_aas = next(iter(test_codon_loader))
#print('next codon train=', tensor_to_codon(next_batch_codons[-1], codonsList))
#print('next aa train=', tensor3_to_sequence(next_batch_aas[-1].unsqueeze(0), aminoAcidLetters))

#print('codon tensor shape=', next_batch_codons.shape)
#print('aa tensor shape=', next_batch_aas.shape)

#count_STOP_codon = 0
#for elem in range(batch_size_of_codons):
    #print(tensor_to_codon(next_batch_codons[elem], codonsList), '\t', tensor3_to_sequence(next_batch_aas[elem].unsqueeze(0), aminoAcidLetters))
#    if tensor3_to_sequence(next_batch_aas[elem].unsqueeze(0), aminoAcidLetters) == '*':
#        count_STOP_codon += 1
#print('number of STOP codons in the previous batch = ', count_STOP_codon)

# neural network architecture: multilayer perceptron (MLP) or fully connected network
# The MLP (fully connected network) will be given as input batches of codons encoded as tensors of dimensions:
# single input tensor = 1 x 64. Or alternatively:
# single input tensor = 1 x 12 (= 3 x (1, 4) where (1, 4) is the OHE per nucleotide x 3 nucleotides in a triplet.
# or with mini-batches having batch size B=batch_size_of_codons --> inputs: B x 1 x 64 (or B x 1 x 12)
# The output is a predicted amino acid determined from the index of the highest probability for the prediction
# of the 21 classes of amino acid (STOP=* sign included). The output comes out of a softmax computed elementwise on a tensor of size
# 1 x 21 mapping the classes. The loss function will be the cross entropy loss. The cross entropy receives 2 arguments:
# argument 1 (predicted output): 21 probabilities (from softmax): tensor of size 1 x 21 with 21 probabilities in the range [0, 1], all summing up to 1.0.
# argument 2 (ground truth): target amino acid tokenized as a one hot encoded tensor of size 1 x 21 (all zeros and a single 1.0 at index of the amino acid).
# note that the predicted amino acid has the index where probability is the highest (argmax of softmax output).

# hyperparameters:
#-----------------
number_of_classes = 21
batch_size = batch_size_of_codons # the number of codon-target amino acid pairs in a batch
input_features = 64 # 64 features per codon (results from One Hot Encoding of the 64 possible different triplets of 4 nucleotide letters)
#input_features = 12 # 12 features per codon (results from One Hot Encoding of the triplets, i.e=3 letters taken from of 4 nucleotide letters alphabet)
output_features = 21
hidden_features_layer1 = 64
hidden_features_layer2 = 512
learning_rate = 0.01
num_epochs = 20 # number of iterations in the training loop
num_heatmap_samples_per_epoch = 3 # we will produce three heatmaps per epoch: 75 heatmaps to be plotted
batches_per_epoch = len(train_codon_set)/batch_size_of_codons
sampling_dist = math.floor(batches_per_epoch/num_heatmap_samples_per_epoch)
samples_id = [int(1*sampling_dist), int(2*sampling_dist), int(3*sampling_dist)]
print('samples id per epoch=', samples_id)

# genetic code full dictionary guess:
#------------------------------------
#genetic_code_array = np.array((21, 64))
genetic_code_array = np.zeros((21, 64)) # initializing with full uncertainty
# this dictionary will be updated during training...

# Defining the MLP architecture:
#-------------------------------
class MLP(nn.Sequential): # Multilayer Perceptron Network
    def __init__(self, input_features, output_features, hidden_features_layer1, hidden_features_layer2):
        super().__init__(
            nn.Flatten(),
            nn.Linear(input_features, hidden_features_layer1, bias=False),
            #nn.ReLU(), # could be removed (no activation between the two layers could be possible as well)
            nn.Linear(hidden_features_layer1, hidden_features_layer2),
            nn.Linear(hidden_features_layer2, hidden_features_layer2),
            nn.Linear(hidden_features_layer2, hidden_features_layer2),
            nn.Linear(hidden_features_layer2, hidden_features_layer2),
            nn.Linear(hidden_features_layer2, hidden_features_layer2),
            nn.Linear(hidden_features_layer2, hidden_features_layer2),
            #nn.ReLU(), # # could be removed (no activation between the two layers could be possible as well)
            #nn.Tanh(), # could use other activation functions
            #nn.Sigmoid(), # could use other activation functions
            nn.Linear(hidden_features_layer2, output_features, bias=False),
            #nn.Softmax(dim=1), # --> here we produce 21 probabilities mapping the 21 classes
            # nn.LogSoftmax(dim=1)  # if NLL would be used as criterion
        )

network = MLP(input_features, output_features, hidden_features_layer1, hidden_features_layer2)
# network is the instantiated mlp from the class MLP:
# transfer MLP network to GPU, just once
device = 'cpu'
network.to(device)

# loss function:
# criterion: (crossentropy loss - multinomial case with 21 classes - adjusting for unbalanced amino acid distribution):
# to adjust for unbalanced amino acid distribution in training set and test set: move to device:
aaWeight = aaWeight.to(device)
criterion = nn.CrossEntropyLoss(weight=aaWeight)
#criterion = nn.CrossEntropyLoss(weight=None)

# optimizer:
#optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
#optimizer = torch.optim.RMSprop(network.parameters(), lr=learning_rate)

# Training the network
#---------------------
def train(num_epochs):
    training_accuracy = 0.0
    train_avg_loss = []
    test_avg_loss = []
    training_accuracy_list = []
    test_accuracy_list = []
    training_pairs_list = []
    elapsed_time_at_epoch = []
    t0 = time.time_ns()  # nanoseconds / start timer
    for i in range(num_epochs):
        train_losses = []
        test_losses = []

        correct = 0
        training_pairs = 0
        batch_iter = 0
        for x, y in train_codon_loader:
            # If the batch size is 64 and with 512,000 pairs in training set,
            # there will be 8,000 batches per epoch.
            batch_iter += 1
            # transfer each minibatch on the GPU:
            # forward pass on GPU:
            x, y = x.to(device), y.to(device)
            training_pairs += x.size()[0]
            pred = network(x)
            loss = criterion(pred, y.squeeze(1))
            train_losses.append(loss.detach())  # better than leaking memory to cpu...

            ###### !!!!!!!!! #######
            pred = nn.functional.softmax(pred)
            y_pred = torch.argmax(pred, dim=1)
            ###### !!!!!!!!! #######

            #y_pred = pred.argmax(dim=1)
            #this_t = index_to_tensor(y_pred, aminoAcidLetters)
            #print('the predicted amino acid were:', tensor3_to_sequence(this_t, aminoAcidLetters))
            #print('the ground truth amino acid were:', tensor3_to_sequence(y, aminoAcidLetters))
            y_pred = index_to_tensor(y_pred, aminoAcidLetters).to(device)
            matches = y_pred*y
            batch_accuracy = matches.sum()
            batch_accuracy = batch_accuracy/batch_size_of_codons
            correct += matches.sum()
            #correct += (y_pred.squeeze(1) == y.squeeze(1)).sum()
            #print('correct training accu of this batch =', batch_accuracy, '\t', format(batch_accuracy/batch_size_of_codons, '4.2%'))
            # produce heatmap when required:
            if batch_iter in samples_id:
                elapsed_time = time.time_ns()  # nanoseconds
                deltaTime = (elapsed_time - t0) / (60e9)  # minutes
                nb_codons_presented = max(0, (i - 1) * batches_per_epoch * batch_size_of_codons) + batch_iter * batch_size_of_codons
                # last_training_accuracy = min(batch_accuracy, training_accuracy)
                # updating the genetic code likelihood with current batch data and current best prediction
                for batch_codon in range(x.size()[0]):
                    #print('x[batch_codon].size()', x[batch_codon].size())
                    #print('x[batch_codon]', x[batch_codon])
                    #print('x[batch_codon, 0, :].tolist().index(1.0)', x[batch_codon, 0, :].tolist().index(1.0))
                    col = x[batch_codon, 0, :].tolist().index(1.0)  # column index of codon
                    for row in range(pred.size()[1]):  # for each of the  21 target amino acids
                        #print('pred.size()', pred.size())
                        #print('pred[batch_codon]', pred[batch_codon])
                        # print('probability', pred[batch_codon, row])
                        genetic_code_array[row][col] = pred[batch_codon, row]
                # update and save heatmap:
                updateANDsave_heatmap(i, batch_iter, sampling_dist, genetic_code_array,
                                      deltaTime, nb_codons_presented, training_accuracy, FIGS_PATH, FIG_LABEL)
            # backprop with automatic differentiation:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_accuracy = correct / len(train_codon_set)
        training_accuracy = training_accuracy.to('cpu')
        print('training accuracy =', training_accuracy, '\t', format(training_accuracy, '4.2%'))

        with torch.no_grad():
            correct = 0

            for x, y in test_codon_loader:
                # transfer each minibatch on the GPU:
                x, y = x.to(device), y.to(device)

                pred = network(x)
                loss = criterion(pred, y.squeeze(1))
                # test_losses.append(loss)
                test_losses.append(loss.detach())

                ###### !!!!!!!!! #######
                pred = nn.functional.softmax(pred)
                pred_max = torch.argmax(pred, dim=1)
                ###### !!!!!!!!! #######

                y_pred = index_to_tensor(pred_max, aminoAcidLetters).to(device)
                matches = y * y_pred
                batch_accuracy = matches.sum()
                correct += matches.sum()
                #correct += (y_pred.squeeze(1) == y.squeeze(1)).sum()
                # print('correct test accum of this batch =', batch_accuracy/batch_size_of_codons)

            test_accuracy = correct / len(test_codon_set)

            # accuracy.detach() # ??
            test_accuracy = test_accuracy.to('cpu')
            print('test accuracy =', test_accuracy, '\t', format(test_accuracy, '4.2%'))
        print('number of pairs of codons-amino acid presented for training = ', training_pairs)
        training_pairs_list.append(training_pairs)
        elapsed_time = time.time_ns()  # nanoseconds
        elapsed_time_at_epoch.append((elapsed_time - t0)/1e9) # in seconds
        # Training score and test score metrics monitoring:
        if (i % 1 == 0 or i == num_epochs - 1):
            print('iteration = ', i, '\t', 'progression =', format(i / num_epochs, '3.1%'))
            print('mean of train losses =', sum(train_losses) / len(train_losses), '\t', 'mean of test losses=',
                  sum(test_losses) / len(test_losses))
            print('training accuracy = ', format(training_accuracy, '4.2%'), 'test accuracy = ', format(test_accuracy, '4.2%'))
        train_avg_loss.append((sum(train_losses) / len(
            train_losses)).item())  # we take the average loss for each minibatch in the training set
        test_avg_loss.append(
            (sum(test_losses) / len(test_losses)).item())  # we take the average loss for each minibatch in the test set
        training_accuracy_list.append(training_accuracy)
        test_accuracy_list.append(test_accuracy)

    return train_avg_loss, test_avg_loss, training_accuracy_list, test_accuracy_list, elapsed_time_at_epoch, training_pairs_list

# training the network and monitor scoring metrics both on training set and test set:
train_avg_loss, test_avg_loss, training_accuracy_list, test_accuracy_list, elapsed_time_at_epoch, training_pairs_list = train(num_epochs)

# training heatmap
# plot dynamically the heatmap of the predicted genetic code with color bar representing the probability that a codon maps
# the target amino acid:

# Test score metrics:
# test accuracy

# Plot the scoring metrics:
fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

axs[0].set_title('Training batch averaged Loss')
axs[0].plot(train_avg_loss)
#axs[0].set_xlabel('Iterations')
axs[0].set_ylabel('Cross Entropy\n Loss')
#yMax = max(train_avg_loss)*1.10
#axs[0].set_ylim(0, yMax)

axs[1].set_title('Test batch averaged Loss')
axs[1].plot(test_avg_loss)
#axs[1].set_xlabel('Iterations')
axs[1].set_ylabel('Cross Entropy \nLoss')
#yMax = max(test_avg_loss)*1.10
#axs[1].set_ylim(0, yMax)

axs[2].set_title('Training Accuracy')
axs[2].plot(training_accuracy_list)
#axs[2].set_xlabel('Iterations')
axs[2].set_ylabel('Accuracy')
axs[2].set_ylim(0.0, 1.0)

axs[3].set_title('Test Accuracy')
axs[3].plot(test_accuracy_list)
axs[3].set_xlabel('Iterations')
axs[3].set_ylabel('Accuracy')
axs[3].set_ylim(0.0, 1.0)
#plt.tight_layout()
#plt.show()
#save figure:
#plt.savefig(r"C:\Users\marcjoiret\Desktop\MARC\TISSUE ENGINEERING and REGENERATIVE MEDICINE\DEEP LEARNING\Project\FIG\MLP_40iter64bitsWeights.svg")
#plt.savefig(r"C:\Users\marcjoiret\Desktop\MARC\TISSUE ENGINEERING and REGENERATIVE MEDICINE\DEEP LEARNING\Project\FIG\MLP_40iter64bitsWeights.pdf")

plt.savefig(f"./{FIGS_PATH}/MLP_40iter64bits1024weights.svg")
plt.savefig(f"./{FIGS_PATH}/MLP_40iter64bits1024weights.pdf")

# This produce a file saving the losses and train + test accuracies for each iteration
#-------------------------------------------------------------------------------------
filePATH = FIGS_PATH
ThisFileName = "MLP64bitsLayer1024weights.txt"
scoresfile = open(filePATH+ThisFileName, 'w')
headerline = 'MLP OHE 64 bits Hidden Layer 1024 with 40 epochs with adjustment for classes weights.\n'
scoresfile.write(headerline) # write header line just once
headerline = 'epoch' + '\t' + 'train_loss' + '\t' + 'test_loss' + '\t' + 'train_accuracy' + '\t' + 'test_accuracy' + '\n'
scoresfile.write(headerline)
for i in range(len(train_avg_loss)):
    line = format(i+1,'d') + '\t' + format(train_avg_loss[i], '5.3f') + '\t' + format(test_avg_loss[i], '5.3f')+\
           '\t' + format(training_accuracy_list[i], '5.2f') +'\t'+ format(test_accuracy_list[i], '5.2f') +'\n'
    scoresfile.write(line)
scoresfile.close()


print('number of epochs = ', num_epochs)
print('elapsed time at each epoch in seconds', elapsed_time_at_epoch)
print('pairs of codons-amino acid presented for training at each epoch', training_pairs_list)
print('training accuracy at each epoch', training_accuracy_list)
print('test accuracy at each epoch', test_accuracy_list)
sys.exit()