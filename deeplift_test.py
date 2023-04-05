import keras
from tensorflow.python.ops.gen_math_ops import select
import deeplift
from deeplift.conversion import kerasapi_conversion as kc
from deeplift.layers import NonlinearMxtsMode
from deeplift.util import get_shuffle_seq_ref_function
from deeplift.dinuc_shuffle import dinuc_shuffle 
from tensorflow.keras.models import load_model
import random 
import seaborn as sns
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


'''
sample_path: the path of samples (fasta)
species: one of three options (Arabidopsis_models/Oryza_sativa_models/Zea_mays_models)
tf: the name of TF

'''

def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
        [0.0, 0.0],
        [0.5, 1.0],
        [0.5, 0.8],
        [0.2, 0.0],
        ]),
        np.array([
        [1.0, 0.0],
        [0.5, 1.0],
        [0.5, 0.8],
        [0.8, 0.0],
        ]),
        np.array([
        [0.225, 0.45],
        [0.775, 0.45],
        [0.85, 0.3],
        [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1,height])[None,:]*polygon_coords
                                                + np.array([left_edge,base])[None,:]),
                                                facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))


def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.825, base+0.085*height], width=0.174, height=0.415*height,
                                            facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.625, base+0.35*height], width=0.374, height=0.15*height,
                                            facecolor=color, edgecolor=color, fill=True))


def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.4, base],
                width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base+0.8*height],
                width=1.0, height=0.2*height, facecolor=color, edgecolor=color, fill=True))

default_colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}
default_plot_funcs = {0:plot_a, 1:plot_c, 2:plot_g, 3:plot_t}
def plot_weights_given_ax(ax, array,
                height_padding_factor,
                length_padding,
                subticks_frequency,
                highlight,
                colors=default_colors,
                plot_funcs=default_plot_funcs):
    if len(array.shape)==3:
        array = np.squeeze(array)
    assert len(array.shape)==2, array.shape
    if (array.shape[0]==4 and array.shape[1] != 4):
        array = array.transpose(1,0)
    assert array.shape[1]==4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        #sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i,:]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color=colors[letter[0]]
            if (letter[1] > 0):
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]                
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)


    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos,min_depth],
                    width=end_pos-start_pos,
                    height=max_height-min_depth,
                    edgecolor=color, fill=False))
        
    ax.set_xlim(-length_padding, array.shape[0]+length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0]+1, subticks_frequency))
    height_padding = max(abs(min_neg_height)*(height_padding_factor),
                        abs(max_pos_height)*(height_padding_factor))
    ax.set_ylim(min_neg_height-height_padding, max_pos_height+height_padding)


def plot_weights(array,tf,idx,
                figsize=(20,2),
                height_padding_factor=0.2,
                length_padding=1.0,
                subticks_frequency=1.0,
                colors=default_colors,
                plot_funcs=default_plot_funcs,
                highlight={}):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(311) 
    plot_weights_given_ax(ax=ax, array=array,
        height_padding_factor=height_padding_factor,
        length_padding=length_padding,
        subticks_frequency=subticks_frequency,
        colors=colors,
        plot_funcs=plot_funcs,
        highlight=highlight)
    plt.ylabel('DeepLIFT\ncontribution\nscore')

    ax = fig.add_subplot(312)
    plt.xlim(0,500)
    plt.ylabel('In-silico\ntiling deletion')
    plt.bar(x=range(len(array[1])),height=array[1],color='salmon',width=5,edgecolor='k',linewidth=0.5)
    
    
    ax = fig.add_subplot(313)
    plt.xlim(0,500)
    # plt.ylabel('In-silico\nmutagenesis')
    sns.heatmap(array[2], cmap='Blues_r', cbar_kws={'location': 'top','fraction':0.1} )
    plt.ylabel('In-silico\nmutagenesis')

    plt.savefig('./%s_deeplift/%s.svg'%(tf,idx), format='svg')
    plt.show()


def population_mutator(population_current , sequence_length) :
#     population_current = population_remove_flank(population_current)
    population_next = []  
    for i in range(len(population_current)) :         
        for j in range(sequence_length) : 
        #First create three copies of the same individual, one for each possible mutation at the basepair.
            population_next.append(list(population_current[i]))
            population_next.append(list(population_current[i]))
            population_next.append(list(population_current[i]))
            population_next.append(list(population_current[i]))

            if (population_current[i][j] == 'A') :
                population_next[4*(sequence_length*i + j) ][j] = 'A'
                population_next[4*(sequence_length*i + j) + 1 ][j] = 'C'
                population_next[4*(sequence_length*i + j) + 2][j] = 'G'
                population_next[4*(sequence_length*i + j) + 3][j] = 'T'
                
            elif (population_current[i][j] == 'C') :
                population_next[4*(sequence_length*i + j)][j] = 'A'
                population_next[4*(sequence_length*i + j) + 1][j] = 'C'
                population_next[4*(sequence_length*i + j) + 2][j] = 'G'
                population_next[4*(sequence_length*i + j) + 3][j] = 'T'
            elif (population_current[i][j] == 'G') :
                population_next[4*(sequence_length*i + j)][j] = 'A'
                population_next[4*(sequence_length*i + j) + 1][j] = 'C'
                population_next[4*(sequence_length*i + j) + 2][j] = 'G'
                population_next[4*(sequence_length*i + j) + 3][j] = 'T'
            elif (population_current[i][j] == 'T') :
                population_next[4*(sequence_length*i + j)][j] = 'A'
                population_next[4*(sequence_length*i + j) + 1][j] = 'C'
                population_next[4*(sequence_length*i + j) + 2][j] = 'G'
                population_next[4*(sequence_length*i + j) + 3][j] = 'T'
        
#     population_next= population_add_flank(population_next)        
    return list(population_next)

def old_seq2feature(data):
    A_onehot = np.array([1,0,0,0], dtype='float32')
    C_onehot = np.array([0,1,0,0], dtype='float32')
    G_onehot = np.array([0,0,1,0], dtype='float32')
    T_onehot = np.array([0,0,0,1], dtype='float32')
    N_onehot = np.array([0,0,0,0], dtype='float32')

    mapper = {'A':A_onehot,'C':C_onehot,'G':G_onehot,'T':T_onehot,'N':N_onehot}

    transformed = np.asarray(([[mapper[k] for k in (data[i])] for i in (range(len(data)))]))
    return transformed


def calculate(sample_path,species,tf):
    with open(sample_path) as f:
        data = []
        while True:
            line = f.readline()
            if not line:
                break
            if not line[0] == '>':
                data.append(line.strip()) 

    def one_hot_encode_along_channel_axis(sequence):
        to_return = np.zeros((len(sequence),4), dtype=np.int8)
        seq_to_one_hot_fill_in_array(zeros_array=to_return,
                                    sequence=sequence, one_hot_axis=1)
        return to_return

    def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
        assert one_hot_axis==0 or one_hot_axis==1
        if (one_hot_axis==0):
            assert zeros_array.shape[1] == len(sequence)
        elif (one_hot_axis==1): 
            assert zeros_array.shape[0] == len(sequence)
        #will mutate zeros_array
        for (i,char) in enumerate(sequence):
            if (char=="A" or char=="a"):
                char_idx = 0
            elif (char=="C" or char=="c"):
                char_idx = 1
            elif (char=="G" or char=="g"):
                char_idx = 2
            elif (char=="T" or char=="t"):
                char_idx = 3
            elif (char=="N" or char=="n"):
                continue #leave that pos as all 0's
            else:
                raise RuntimeError("Unsupported character: "+str(char))
            if (one_hot_axis==0):
                zeros_array[char_idx,i] = 1
            elif (one_hot_axis==1):
                zeros_array[i,char_idx] = 1

    selected_path= './DenseNet_models/'+species+'/'+tf+'_checkmodel.hdf5'
    deeplift_model =\
        kc.convert_model_from_saved_files(
            selected_path,
            nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)
    l=list(deeplift_model.get_name_to_layer().keys())


    deeplift_contribs_func = deeplift_model.get_target_contribs_func(
                                find_scores_layer_name=l[0],
                                pre_activation_target_layer_name=l[-2])
    

    rescale_conv_revealcancel_fc_many_refs_func = get_shuffle_seq_ref_function(
        #score_computation_function is the original function to compute scores
        score_computation_function=deeplift_contribs_func,
        #shuffle_func is the function that shuffles the sequence
        #technically, given the background of this simulation, randomly_shuffle_seq
        #makes more sense. However, on real data, a dinuc shuffle is advisable due to
        #the strong bias against CG dinucleotides
        shuffle_func=dinuc_shuffle,
        one_hot_func=lambda x: np.array([one_hot_encode_along_channel_axis(seq) for seq in x]))

    num_refs_per_seq=10 #number of references to generate per sequence

    scores_without_sum_applied = rescale_conv_revealcancel_fc_many_refs_func(
                task_idx=0, #can provide a list of tasks; references will be reused for each
                #Providing a single integeter for task_idx works too and would return a numpy array rather
                # than a list of numpy arrays
                input_data_sequences=data,
                num_refs_per_seq=num_refs_per_seq,
                batch_size=200,
                progress_update=1000)
    scores=np.sum(scores_without_sum_applied,axis=2)
    if not os.path.exists('./%s_deeplift'%(tf)):
            os.mkdir('./%s_deeplift'%(tf))
    # np.save('./%s_deeplift/scores.npy'%tf,scores)

    onehot_data = np.array([one_hot_encode_along_channel_axis(seq) for seq in data])
    model= load_model(selected_path)
    for idx in range(len(data)):
        print("Scores for example",idx)
        scores_for_idx = scores[idx]
        original_onehot = onehot_data[idx]
        scores_for_idx = original_onehot*scores_for_idx[:,None]
        

        predict_probabilty=model.predict(onehot_data)[0][0]
        preds = []
        for idx in range(len(data)):
            for i in range(0,len(data[idx])-10+1, 1):
                # temp = seq2onehot[]
                temp = np.vstack((onehot_data[0][:i,:],onehot_data[0][i+10:,:]))
                seq = np.pad(temp,((0,10),(0,0)),'constant',constant_values=(0,0))[np.newaxis,:,:]
                predict = model.predict(seq)[0]
                preds.append(predict[0])
            # predict_probabilty=str(predict_probabilty[0][0])
            diff_deletion = [i - predict_probabilty for i in preds]


            #calculate mutation
            population_1bp_all_sequences = population_mutator(data, 500)
            # for i in range(0,len(population_1bp_all_sequences), len(population_1bp_all_sequences)):
            population_1bp_all_feature = old_seq2feature(population_1bp_all_sequences)
            population_1bp_fitness = model.predict(population_1bp_all_feature)
            diff_mutation_o = population_1bp_fitness - predict_probabilty
            diff_mutation = np.reshape(diff_mutation_o,[500,4]).T
            plot_weights([scores_for_idx, diff_deletion, diff_mutation], tf=tf,idx=idx,subticks_frequency=20,figsize=(50,4))


def main():
    sample_path=sys.argv[1]
    species=sys.argv[2]
    tf=sys.argv[3]
    calculate(sample_path,species,tf)
    
if __name__ == "__main__":
    main()
