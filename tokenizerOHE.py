import io
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

t_start = time.time_ns()

# load the the datasets (files): input data file and output data (target) file:
def loadData(input_filename, target_filename):
    # read the file and split into lines and save the index of the line:
    def read_lines(filename):
        # this will return the list of the cdsID unique keys associated to the sequence (or index)
        # and the list of the sequences in the file:
        # open the file and skip the (first) header line:
        lines = io.open(filename, 'r').read().split('\n')
        stringCol = [line.split('\t') for line in lines]
        NB_seq = len(stringCol)
        # (the headerline is skipped by starting at list index 1 instead of zero)
        return [stringCol[i][0] for i in range(1, NB_seq-1)], [stringCol[i][1] for i in range(1, NB_seq-1)]

    cds_id_mRNA_list, transcript_sequences_list = read_lines(input_filename)
    cds_id_prot_list, protein_sequences_list = read_lines(target_filename)
    return cds_id_mRNA_list, transcript_sequences_list, cds_id_prot_list, protein_sequences_list

#filePATH = r'C:\Users\marcjoiret\Desktop\MARC\TISSUE ENGINEERING and REGENERATIVE MEDICINE\DEEP LEARNING\Project\DATA\'
filePATH = r"../DATA/"
fileOne = r"humanORFs.txt"
fileTwo = r"humanProteins.txt"
cdsIDmRNA, ORFs, cdsIDprot, proteins = loadData(filePATH+fileOne, filePATH+fileTwo)

# sanitary check:
print('First few cds id in transcripts list and protein list')
for i in range(5):
    print('#' + '\t' + 'cdsID' + '\t' + 'cdsID')
    print(str(i+1) + '\t' + str(cdsIDmRNA[i]) + '\t' + str(cdsIDprot[i]))
    print('mRNA:')
    print(ORFs[i])
    print('protein:')
    print(proteins[i])

print('Number of ORFS=', len(ORFs))
print('Number of proteins=', len(proteins))

# sanitary check:
# the cds id in each file should match exactly. Let us check it is indeed the case:
cdsmRNAArray = np.array(cdsIDmRNA)
cdsProtArray = np.array(cdsIDprot)
matchingArray = np.array(cdsProtArray==cdsmRNAArray)
print('The number of matching should be equal to the number of sequences in both files =', matchingArray.sum()==len(cdsIDmRNA))
t_stop = time.time_ns()
print('Elapsed time for data loading = ', format((t_stop-t_start)/1e6, '8.2f'), ' milliseconds.')

# in the two input files, we have either A, U, C, G characters in the mRNA language, i.e. 4 characters, or
# the 20 amino acid one letter code + '*' as the character mapped by a STOP codon,
# i.e., 21 characters in the protein language.

# characters (alphabet) of the transcripts language (mRNA):
nucleotideLetterList = ['A', 'U', 'C', 'G']
NB_nucleotideLETTERS = len(nucleotideLetterList) # n_letters in the nucleic acid alphabet
nucleotideLetters = "AUCG"

# characters (alphabet) of the proteins language:
aminoAcidOneLetterList = ['*', 'G', 'V', 'K', 'N', 'Q', 'H', 'E', 'D', 'Y', 'C', 'F', 'I', 'M', 'W', 'R', 'L', 'S', 'T', 'P', 'A']
NB_aaLETTERS = len(aminoAcidOneLetterList)  # n_letters in the protein alphabet
aminoAcidLetters ="*GVKNQHEDYCFIMWRLSTPA"

# aminoAcidWeights: the distribution of amino acids in proteins is unbalanced (not uniform):
#aminoAcidWeights = np.array([6.62, 6.09, 5.65, 3.62, 4.66, 2.61, 6.88, 4.71, 2.76, 2.33, 3.80, 4.44, 2.21, 1.32, 5.69, 10.05, 8.14, 5.34, 6.13, 6.95])
aminoAcidWeights = np.array([0.34, 6.597492,  6.069294,  5.63079,   3.607692,  4.644156,  2.601126,  6.856608,
  4.693986,  2.750616,  2.322078,  3.78708,   4.424904,  2.202486,  1.315512,
  5.670654, 10.01583,   8.112324,  5.321844,  6.109158,  6.92637 ])
aminoAcidWeights = 1.0/aminoAcidWeights
aaWeight = torch.tensor(aminoAcidWeights)

# codons (triplets list): in mRNA language: UCAG
codonsList = ['UUU', 'UUC', 'UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG', 'AUU', 'AUC', 'AUA', 'AUG', 'GUU', 'GUC', 'GUA', 'GUG',
              'UCU', 'UCC', 'UCA', 'UCG', 'CCU', 'CCC', 'CCA', 'CCG', 'ACU', 'ACC', 'ACA', 'ACG', 'GCU', 'GCC', 'GCA', 'GCG',
              'UAU', 'UAC', 'UAA', 'UAG', 'CAU', 'CAC', 'CAA', 'CAG', 'AAU', 'AAC', 'AAA', 'AAG', 'GAU', 'GAC', 'GAA', 'GAG',
              'UGU', 'UGC', 'UGA', 'UGG', 'CGU', 'CGC', 'CGA', 'CGG', 'AGU', 'AGC', 'AGA', 'AGG', 'GGU', 'GGC', 'GGA', 'GGG']

colorList = ['orange', 'orange', 'green', 'green', 'green', 'green', 'green', 'green', 'gray', 'gray', 'gray', 'black', '#FF00FF', '#FF00FF', '#FF00FF', '#FF00FF',
             '#603101', '#603101','#603101','#603101', '#ad0f6c', '#ad0f6c', '#ad0f6c', '#ad0f6c', '#16537E', '#16537E', '#16537E', '#16537E', '#F6CA4C', '#F6CA4C', '#F6CA4C', '#F6CA4C',
             '#008080', '#008080', '#858B5F', '#858B5F', '#7289DA', '#7289DA', 'pink', 'pink', '#703D6E', '#703D6E', '#ff5151', '#ff5151', '#3577dc', '#3577dc', '#22499e', '#22499e',
             '#999999', '#999999', '#858B5F', '#ff6600', '#e70005', '#e70005', '#e70005', '#e70005', '#603101', '#603101', '#e70005', '#e70005', '#1cc48a', '#1cc48a', '#1cc48a', '#1cc48a']

# codons dictionary of index (possibly used for codons embeddings):
codon_to_idx = {triplet: i for i, triplet in enumerate(codonsList)}

# To represent a single letter, we use a “one-hot vector” of
# size <1 x n_letters>. A one-hot vector is filled with 0s
# except for a 1 at index of the current letter, e.g. "U" = <0 1 0 0 > in the nucleotide alphabet
# or '*' = <1 0 0 0 0 0 0 0 ...> in the amino acid alphabet.

# To make a N-gram or word or sequence we join a bunch of those into a
# 2D matrix <line_length x 1 x n_letters>.

# That extra 1 dimension is because PyTorch assumes
# everything is in batches - we will just use a batch size of 1 here.
# In our project, a batch size of 1, mean we take one (line) sample at a time, i.e., one mRNA sequence at a time.

# Find letter index from all_letters, e.g. "A" = 0
def letter_to_index(letter, alphabet):
    return alphabet.find(letter)

# Just for demonstration, turn a letter into a 1 by n_letters Tensor
def letter_to_tensor(letter, alphabet):
    tensor = torch.zeros(1, len(alphabet))
    tensor[0][letter_to_index(letter, alphabet)] = 1
    return tensor

# Turn a sequence into a sequence_length x 1 x n_letters,
# or an array of one-hot letter vectors
def sequence_to_tensor(line, alphabet):
    tensor = torch.zeros(len(line), 1, len(alphabet))
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter, alphabet)] = 1
    return tensor

# Turn a codon (letter triplet) into a tensor 1 x 64 (One Hot Encoding of codons as tensors of size 1 x 64):
def codon_to_tensor(codon, codonsList):
    tensor = torch.zeros(1, len(codonsList))
    tensor[0][codonsList.index(codon)] = 1
    return tensor

# let's check how this goes...
#thisCodon = codonsList[1] # =UUC
#outTensor = codon_to_tensor(thisCodon, codonsList)
#print(outTensor.shape)
#print(outTensor)
#print('Tadaaaa 64')

# Turn a codon (letter triplet) into a tensor 1 x 12 (= 3 times a one letter One Hot Encoding of tensors of size 1 x 4)
def codon_to_tensor_12bits(thisCodon, alphabet):
    # codon is a string of 3 letters. Convert each letter to a tensor of size 1 x 4,
    # then concatenate the 3 tensors just obtained to return a correct view of size 1 x 12 of
    # the 'stacked tensor that was temporarily 3 x 1 x 4
    t0_of_4 = letter_to_tensor(thisCodon[0], alphabet)
    t1_of_4 = letter_to_tensor(thisCodon[1], alphabet)
    t2_of_4 = letter_to_tensor(thisCodon[2], alphabet)
    # return a tensor of size 1 x 12:
    return torch.stack((t0_of_4, t1_of_4, t2_of_4), dim=0).view(-1)

# let's check how this goes...
#thisCodon = codonsList[1] # =UUC
#outStackedTensor = codon_to_tensor_12bits(thisCodon, nucleotideLetters)
#print(outStackedTensor.shape)
#print(outStackedTensor)
#print('and back to 3 x 1 x 4')
#print(outStackedTensor.view(3, 1, 4))
#print('Tadaaaa 12 bits and 3 x 4')
#sys.exit()

def index_to_tensor(idx_tensor, alphabet):
    # Given the index of an amino acid in the aminoacid alphabet, returns the OHE tensor for this aa (1 x 21):
    tensor = torch.zeros(idx_tensor.size()[0], 1, len(alphabet))
    for elem in range(len(idx_tensor.tolist())):
        tensor[elem, 0, idx_tensor.tolist()[elem]] = 1
    return tensor

def tensor_to_codon(t, codonsList):
    return codonsList[t[0].tolist().index(1.0)]

def tensor3x4_to_codon(t3x4, alphabet):
    triplet = str()
    for letter_idx in range(t3x4.size()[0]):
        triplet += alphabet[t3x4[letter_idx, 0].tolist().index(1.0)]
    return triplet

#nucl_letter = 'U'
#print('nucleotide letter converted to tensor:')
#print('nucleotide letter = ', nucl_letter)
#print('tensor =', letter_to_tensor(nucl_letter, nucleotideLetters))

#aa_letter = 'W'
#print('amino acid letter converted to tensor:')
#print('amino acid letter = ', aa_letter)
#print('tensor =', letter_to_tensor(aa_letter, aminoAcidLetters))
#aa_letter = '*'
#print('amino acid letter = ', aa_letter)
#print('tensor =', letter_to_tensor(aa_letter, aminoAcidLetters))

#sequence_mRNA = 'AAAUUCGGG'
#print(sequence_mRNA)
#print(sequence_to_tensor(sequence_mRNA, nucleotideLetters))
#print(sequence_to_tensor(sequence_mRNA, nucleotideLetters).size())  # [9, 1, 4]

#sequence_protein = 'MGWWR*'
#print(sequence_protein)
#print(sequence_to_tensor(sequence_protein, aminoAcidLetters))
#print(sequence_to_tensor(sequence_protein, aminoAcidLetters).size())  # [6, 1, 21]

#ThisCodon = 'GGA'
#print('this codon=', ThisCodon)
#print('codon tensor = ', codon_to_tensor(ThisCodon, codonsList))
#print('codon tensor shape=', codon_to_tensor(ThisCodon, codonsList).size())

# This function produces a heatmap and save the figure in an automatically given named
# file in the specified directory. A number of arguments are required to proper build and
# correctly annotating the file (captioning):
# The fucntion is called upon from the neural network training loop

def updateANDsave_heatmap(this_epoch, this_batch_ID, sampling_D, this_table, this_time, Ncod, last_training_accuracy, figurePATH="./FIGS/", txt_label=""):
    # training heatmap
    # plot dynamically the heatmap of the predicted genetic code with color bar representing the probability that a codon maps
    # the target amino acid:
    # Save a single .pdf or .svg heatmap per epoch with relevant annotations.
    # The heatmap files could later be merged in a gif animation.
    fig_num = str(format(this_epoch, '2d'))
    batch_num=str(int(this_batch_ID/sampling_D))
    if str(fig_num[0])==' ':
        fig_num = str('0')+fig_num[1]
    figNamePNG = "heatmapCode"+fig_num+batch_num+".png"
    # figNameSVG = "heatmapCode"+fig_num+batch_num+".svg"
    print(figNamePNG)
    # print(figNameSVG)
    # create the heatmap with the 2D data: columns = codonList, rows= amino acid list.
    # Each column comes from the last updated probability distribution for the 21 targets amino acid to be mapped by a given codon
    # get the codons and their index from last batch of 64 codons in tensor x[64, 1, 64] for current epoch --> indices in codonList
    # get the 21 updated probabilities for the above 64 codons to map the 21 target amino acid: pred[64, 1, 21]
    # --> assign the 21 probabilities at the right column = codon and at the right aa place (aminoAcidList).

    # Build the heatmap:
    #genetic_code_table = np.random.rand(21, 64)
    genetic_code_table = this_table
    #genetic_code_table[0][3]= 0.99

    fig, ax = plt.subplots(1, 1, figsize=(20, 7))
    my_colors=[(0.2,0.3,0.3),(0.4,0.5,0.4),(0.1,0.6,0),(0.1,0.8,0), (0.0, 1.0, 0)]
    # label the rows and columns:
    ['*', 'G', 'V', 'K', 'N', 'Q', 'H', 'E', 'D', 'Y', 'C', 'F', 'I', 'M', 'W', 'R', 'L', 'S', 'T', 'P', 'A']
    AA_ordered_list = ['STOP', 'Gly G', 'Val V', 'Lys K', 'Asp N', 'Gln Q', 'His H', 'Glu E', 'Asp D', 'Tyr Y',
                       'Cys C', 'Phe F', 'Ile I', 'Met M', 'Trp W', 'Arg R', 'Leu L', 'Ser S', 'Thr T', 'Pro P',
                       'Ala A']
    #row_labels = aminoAcidOneLetterList
    row_labels = AA_ordered_list
    col_labels = codonsList
    ax = sns.heatmap(genetic_code_table, vmin=0.00, vmax=0.07, cmap= my_colors, square=True,
                linewidth=0.2,
                #linecolor=(0.1,0.2,0.2),
                xticklabels=col_labels,
                yticklabels=row_labels,
                ax=ax,
                #center=(vmax-vmin)/2, #??
                annot=False, fmt='.2f',
                annot_kws={'fontsize':16,
                           'color':'white',
                            #'fontweight':'bold',
                            #'fontfamily': 'serif',
                           },
                cbar_kws={"orientation": "vertical"}
                )
    # set up colorbar and display colorbar vertically or horizontally:
    colorbar = ax.collections[0].colorbar
    M=0.07
    colorbar.set_ticks([0.10*M, 0.30*M, 0.50*M, 0.70*M, 0.90*M])
    #colorbar.set_ticklabels(['p < 0.20','p < 0.40','p < 0.60', 'p < 0.80', 'p > 0.80'])
    colorbar.set_ticklabels(['p < 0.014', 'p < 0.028', 'p < 0.042', 'p < 0.056', 'p > 0.056'])
    # set title and suptitle:
    txt_caption = 'EPOCH = '+str(format(this_epoch, '2d'))+'.BATCH = '+str(format(this_batch_ID, '4d')) + ' ELAPSED TIME = '+ str(format(this_time, '3.1f'))+ str(' min. ') +\
                            str(format(int(Ncod), '8d'))+ str(' PAIRS PRESENTED. ') + ' TRAIN. ACCURACY = '+str(format(last_training_accuracy, '4.2%'))
    ax.set_title(label=txt_caption, fontsize=16)
    #{'fontsize':22, 'fontweight':'bold'})
    # rotate the codon names by 45 degrees:
    plt.xticks(rotation = 45)
    plt.yticks(rotation = 0)
    # txt_label = "Genetic Code Deciphering with a RNN 2 layers hidden size 256, adjusting for weights."
    plt.text(x=0.10, y=0.90, s= txt_label, fontsize=24, ha="left", transform=fig.transFigure)

    #plt.show()
    # save figure:
    #figurePATH = r"C:/Users/marcjoiret/Desktop/MARC/TISSUE ENGINEERING and REGENERATIVE MEDICINE/DEEP LEARNING/Project/FIG/MLP/pdf/"
    #figurePATH = r"C:/Users/marcjoiret/Desktop/MARC/TISSUE ENGINEERING and REGENERATIVE MEDICINE/DEEP LEARNING/Project/FIG/MLP64bits/pdf/"
    #figurePATH = r"C:/Users/marcjoiret/Desktop/MARC/TISSUE ENGINEERING and REGENERATIVE MEDICINE/DEEP LEARNING/Project/FIG/MLP12bitsOHEweights/pdf/"
    #figurePATH = r"C:/Users/marcjoiret/Desktop/MARC/TISSUE ENGINEERING and REGENERATIVE MEDICINE/DEEP LEARNING/Project/FIG/MLPembeddings64to10relu_40epByFreqTrue/pdf/"
    #figurePATH = r"C:/Users/marcjoiret/Desktop/MARC/TISSUE ENGINEERING and REGENERATIVE MEDICINE/DEEP LEARNING/Project/FIG/RNN2layersWEIGHTS/pdf/"
    #figurePATH = r"C:/Users/marcjoiret/Desktop/MARC/TISSUE ENGINEERING and REGENERATIVE MEDICINE/DEEP LEARNING/Project/FIG/RNN2layers512/pdf/"
    # figurePATH = r"./FIGS/"

    plt.savefig(figurePATH+figNamePNG)
    #figurePATH = r"C:/Users/marcjoiret/Desktop/MARC/TISSUE ENGINEERING and REGENERATIVE MEDICINE/DEEP LEARNING/Project/FIG/MLP/svg/"
    #figurePATH = r"C:/Users/marcjoiret/Desktop/MARC/TISSUE ENGINEERING and REGENERATIVE MEDICINE/DEEP LEARNING/Project/FIG/MLP64bits/svg/"
    #figurePATH = r"C:/Users/marcjoiret/Desktop/MARC/TISSUE ENGINEERING and REGENERATIVE MEDICINE/DEEP LEARNING/Project/FIG/MLP12bitsOHEweights/svg/"
    #figurePATH = r"C:/Users/marcjoiret/Desktop/MARC/TISSUE ENGINEERING and REGENERATIVE MEDICINE/DEEP LEARNING/Project/FIG/MLPembeddings64to10relu_40epByFreqTrue/svg/"
    #figurePATH = r"C:/Users/marcjoiret/Desktop/MARC/TISSUE ENGINEERING and REGENERATIVE MEDICINE/DEEP LEARNING/Project/FIG/RNN2layersWEIGHTS/svg/"
    #figurePATH = r"C:/Users/marcjoiret/Desktop/MARC/TISSUE ENGINEERING and REGENERATIVE MEDICINE/DEEP LEARNING/Project/FIG/RNN2layers512/svg/"
    # figurePATH = r"./FIGS/"

    # plt.savefig(
        # figurePATH+figNameSVG)

#sys.exit()