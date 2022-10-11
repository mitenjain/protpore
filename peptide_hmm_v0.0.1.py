#!/usr/bin/env python

########################################################################
# hmm.py
########################################################################

import sys, string, time, math, glob, os
import argparse, itertools, collections
import numpy
import h5py
import colorsys
import pyximport
pyximport.install(setup_args={'include_dirs':numpy.get_include()})
from yahmm import *
from PyPore.core import Segment
from PyPore.parsers import *
from Fast5Types import *
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import matplotlib.pyplot as plt
import seaborn as sns
from optparse import OptionParser

########################################################################
# This program is designed to perform segmentation in parallel and make 
# decision dynamically on which segment to keep
# This is to be used with tRNA events
# 
#    1->2->3->4->5->6->7->8->9->10
# 
########################################################################

########################################################################
# Create all possible kmers
########################################################################

def kmer_current_map(file):
    # Read kmer current mean, sd, and time measurements
    kmer_current_dict = {}
    kmer_table = open(file, 'r')
    for line in kmer_table:
        line = line.strip().split('\t')
        key = line[0].strip()
        meanCurrent = float(line[1].strip())
        stdevCurrent = float(line[2].strip())
        if not key in kmer_current_dict.keys():
            kmer_current_dict[key] = 0
        kmer_current_dict[key] = [meanCurrent, stdevCurrent]

    kmer_table.close()

    return kmer_current_dict

########################################################################
# HMM Constructor class constructing input-specific HMMs
########################################################################

class HMM_Constructor():

    def __init__(self):
        pass

    def HMM_linear_model(self, kmer_list, kmer_current_dict, model_name=None):
        '''
        This HMM models the segments corresponding to the context and the label
        Each state will have the following transitions:
        1-step forward (expected) - 1 possible transition
        1-step back-slip - 1 possible transition
        1-step forward skip - 1 possible transition
        Self-loop
        '''
        # Create model and add states
        model = yahmm.Model(name=model_name) if not model_name is None \
                                                else yahmm.Model(name='HMM_linear_model')
        previous_skip = None
        previous_short_slip = None
        current_short_slip = None
        previous_mean = 0
        previous_sd = 0
        states_list = []

        for index in range(len(kmer_list)):
        
            # State name, mean and stdev for the kmer 
            kmer = kmer_list[index]
            current_mean = kmer_current_dict[kmer][0]
            current_sd = kmer_current_dict[kmer][1]
            # Transition probabilities for a match state
            # Self-loop to itself
            self_loop = 0.1
            # End from any state, i.e. reach model.end
            end = 0.05
            # Transitions for Drop-off State
            drop = 0.05
            # Transitions for going to Blip State
            blip = 0.05
            blip_self = 0.05
            # Back Slips, short and long
            slip = 0.05 if index > 0 else 0.00
            # Only short backslips possible
            short_slip = slip
            long_slip = 0.0
            # Transitions from silent slip states
            # Short slip from silent short slip state
            step_back = 0.05
            # Skip that accounts for a missed segment
            skip = 0.1
            # Transitions from current skip silent state to the previous match state or 
            # previous silent skip states
            long_skip = 0.05
            # Transitions for Insert state between two neighboring match states
            insert = 0.05 if index > 0 else 0.00
            # Self loop for an insert state
            ins_self = 0.05
            # Transition to the next match state (Forward Transition)
            # Each match state has transitions out to self_loop, end, drop, blip, slip, 
            # skip, insert, re_read, and forward
            forward = 1 - (self_loop + end + blip + slip + skip + insert)
            # Create and Add State
            current_state = yahmm.State(yahmm.NormalDistribution(current_mean, \
                                        current_sd), \
                                        name = 'M_' + kmer + '_' + str(index))
            model.add_state(current_state)

            # Transitions for the match state 
            # Self-loop to itself
            model.add_transition(current_state, current_state, self_loop)
            # The model could end from any match state
            if index < len(kmer_list) - 1:
                model.add_transition(current_state, model.end, end)

            # Each Match State can go to a silent drop-off state, and then to model.end
            drop_off = yahmm.State(None, name = 'S_DROPOFF_' + kmer + '_' + str(index))
            model.add_state(drop_off)
            # Transition to drop_off and back, from drop_off to end
            model.add_transition(current_state, drop_off, drop)
            model.add_transition(drop_off, current_state, 1.0 - blip_self)

            model.add_transition(drop_off, model.end, 1.00)

            # Each Match State can go to a Blip State that results from a voltage blip
            # Uniform Distribution with Mean and Variance for the whole event
            blip_state = yahmm.State(yahmm.UniformDistribution(15.0, 120.0), \
                                            name = 'I_BLIP_' + kmer + '_' + str(index))
            model.add_state(blip_state)
            # Self-loop for blip_staet
            model.add_transition(blip_state, blip_state, blip_self)
            # Transition to blip_state and back
            model.add_transition(current_state, blip_state, blip)
            model.add_transition(blip_state, current_state, 1.0 - blip_self)

            # Short Backslip - can go from 1 to the beginning but favors 1 > ...
            # Starts at state 1 when the first short slip silent state is created
            if index >= 1:
                # Create and add silent state for short slip
                current_short_slip = yahmm.State(None, name = 'B_BACK_SHORT_' + kmer + \
                                                            '_' + str(index))
                model.add_state(current_short_slip)
                # Transition from current state to silent short slip state
                model.add_transition(current_state, current_short_slip, short_slip)
                if index >= 2:
                    # Transition from current silent short slip state to previous 
                    # match state
                    model.add_transition(current_short_slip, states_list[index-1], \
                                            step_back)
                    # Transition from current silent short slip state to previous silent 
                    # short slip state
                    model.add_transition(current_short_slip, previous_short_slip, \
                                            1 - step_back)
                else:
                    model.add_transition(current_short_slip, states_list[index-1], 1.00)

            # Create and Add Skip Silent State
            current_skip = yahmm.State(None, name = 'S_SKIP_' + kmer + '_' + str(index))
            model.add_state(current_skip)
            
            if not previous_skip is None:
                # From previous Skip Silent State to the current Skip Silent State
                model.add_transition(previous_skip, current_skip, long_skip)
                # From previous Skip Silent State to the current match State
                model.add_transition(previous_skip, current_state, 1 - long_skip)

            # From previous match State to the current Skip Silent State
            if index == 0:
                model.add_transition(model.start, current_skip, 1.0 - forward)
            else:
                model.add_transition(states_list[index-1], current_skip, skip)

            # Insert States
            if index > 0:
                # Mean and SD for Insert State
                # Calculated as a mixture distribution
                insert_mean = (previous_mean + current_mean) / 2.0
                insert_sd =  numpy.sqrt(1/4 * ((previous_mean - current_mean) ** 2) \
                                        + 1/2 * (previous_sd ** 2 + current_sd ** 2))
                # Create and Add Insert State
                # Normal Distribution with Mean and Variance that represent 
                # neighboring states                
                insert_state = yahmm.State( yahmm.NormalDistribution(insert_mean, \
                                                insert_sd ), \
                                                name = 'I_INS_' + kmer + '_' + str(index))
                model.add_state(insert_state)
                # Self-loop
                model.add_transition(insert_state, insert_state, ins_self)
                # Transition from states_list[index-1]
                model.add_transition(states_list[index-1], insert_state, insert)
                # Transition to current_state
                model.add_transition(insert_state, current_state, 1.0 - ins_self)

            # Transition to the next match state
            if index == 0:
                # Only transitions from start to skip silent state or first match state
                model.add_transition(model.start, current_state, forward)
            elif index == 1:
                # Since I add match transitions from the previous match state to current
                # match state, I have to make sure the sum of outgoing edges adds to 1.0
                # For index 0, there is no slip, addition of M_0 -> M_1 happens at 1, 
                # which means add this slip probability to the forward transition for M_0
                model.add_transition(states_list[index-1], current_state, forward + slip)
            else:
                model.add_transition(states_list[index-1], current_state, forward)
    
            # Append the current state to states list            
            states_list.append(current_state)

            # Re-assign current states to previous states
            previous_skip = current_skip
            previous_short_slip = current_short_slip if not current_short_slip is None \
                                                        else None
            previous_mean = current_mean
            previous_sd = current_sd

            # Model end case
            if index == len(kmer_list) - 1:
                skip = 0.0
                insert = 0.0
                forward = 1 - (self_loop + end + blip + slip + skip + insert)
                # End cases
                model.add_transition(states_list[index], model.end, forward + end)
                model.add_transition(previous_skip, model.end, 1.00)

        model.bake()
        return model

def model_maker(kmer_current_dict, model_name=None):
    kmer_list = map(str, range(len(kmer_current_dict)))
    model = HMM_Constructor().HMM_linear_model(kmer_list, kmer_current_dict, \
                                                model_name)
    return model

def prediction(models, sequences, algorithm = 'forward-backward'):

    # Predict sequence from HMM using a user-specified algorithm
    # Forward-Backward (default) or Viterbi
    sequence_from_hmm = []
    for i in range(len(sequences)):
        for model in models:
            if algorithm == 'viterbi':
                sequence_from_hmm.append(model.viterbi(sequences[i]))
            elif algorithm == 'forward-backward':
                sequence_from_hmm.append(model.forward_backward(sequences[i]))
    return sequence_from_hmm

def plot_event(filename, event, model):
    # Plots, top plot is segmented event colored in cycle by segments, bottom 
    # subplot is segmented event aligned with HMM, colored by states
#    plt.figure(figsize=(20, 8))
    plt.figure()
    plt.subplot(211)
    plt.grid()
    event.plot(color='cycle')
    plt.subplot(212)
    plt.grid()
    event.plot(color='hmm', hmm=model, cmap='Set1')
#    plt.show()
    fig_name = filename.strip().split('/')[-1] + '.png'
    plt.tight_layout()
    plt.savefig(fig_name, format='png')
    plt.close()

def plot_event_segments(filename, event):
    # Plots, top plot is segmented event colored in cycle by segments, bottom 
#    plt.figure(figsize=(20, 8))
    plt.figure()
    plt.grid()
    event.plot(color='cycle')
#    plt.show()
    fig_name = filename.strip().split('/')[-1] + '.png'
    plt.tight_layout()
    plt.savefig(fig_name, format='png')
    plt.close()
    

########################################################################
# Main
# Here is the main program
########################################################################

def main(myCommandLine=None):

    t0 = time.time()

    #Parse the inputs args/options
    usageStr = './hmm.py -i ./fast5/ -m ./models/'
    parser = OptionParser(usage=usageStr, version='%prog 0.1')

    #Options
    parser.add_option('-i', dest='fast5', help='fast5 file dir', default='')
    parser.add_option('-m', dest='models', help='models dir', default='./profiles/')
    
    parser.add_option('-n', dest='min', help='min_width', type=int, default=250)
    parser.add_option('-x', dest='max', help='max_width', type=int, default=1000)
    parser.add_option('-w', dest='win', help='window_width', type=int, default=2000)

    #Parse the options/arguments
    options, args = parser.parse_args()

    #Print help message if no input
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    print >> sys.stderr, options

    # fast5 files
    filePath = options.fast5
    # tRNA profiles (index/meancurrent/stddev/time)
    modelPath = options.models
    # Segmentation parameters
    minWidth = options.min
    maxWidth = options.max
    winWidth = options.win

    print 'creating kmer current map'
    # CREATE CURRENT MAPS OUT OF MODELPATH
    # kmer_current_dict[index] = [meancurrent, stddev]
    # one dictionary per file where each entry is one line in the file
    kmer_current_dict_pro001 = kmer_current_map(os.path.join(modelPath, 'pro_001.txt'))

    '''
    Construct models: 
    '''
    # Build one model for each current_dict / filename in modelpath
    # model_maker takes kmer_list, which is just a list of str numbers that are the keys in that dict
    # then model_maker passes the list, dict, name into HMM_linear_model, which gets kmer (just an int),
    # and mean and stddev for each entry in the kmer_dict.
    # Make HMMs for every file and then add them all to a list called models.

    pro_001_model = model_maker(kmer_current_dict_pro001, model_name = 'pro_001')
    models = [pro_001_model]

#    models[0].write(sys.stdout)
    print 'models done'

    # Create blank templates for every model file
    viterbi_prediction = []

    p_pro001 = 0
    accuracy = 0.0

    matrix = {'P_pro001':{}}
#     matrix = {'P_pro001':{}, 'T_Lys':{}, 'T_Phe':{}, 'adpt_DNA_RNA':{}}
    #matrix['T5'] = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    matrix['P_pro001'] = {0:0}
#     matrix['T_Lys'] = {0:0, 1:0, 2:0, 3:0}
#     matrix['T_Phe'] = {0:0, 1:0, 2:0, 3:0}
#     matrix['adpt_DNA_RNA'] = {0:0, 1:0, 2:0, 3:0}

    num_events = 0
    fileCount = 0
    filesets = []

    # Read in each .fast5 file
    for filename in glob.glob(os.path.join(filePath, '*.fast5')):
        fileCount += 1
        f5File = h5py.File(filename, 'r')

        # File will have either Raw read or Events
        try:
            # Case 1
            read = (f5File['/Raw/Channel_1/'])
            # Case 2
#             read = (f5File['/Raw/Channel_5/'])
            signal = (read['Signal']).value
        except:
            print >> sys.stderr, 'Unsupported data type'

        # Find values to be used for converting current to picoAmperes
        # Create current - numpy array of floats in units pA
        digitisation = (read['Meta']).attrs['digitisation']
        offset = (read['Meta']).attrs['offset']
        f5range = (read['Meta']).attrs['range']
        sampling_rate = (read['Meta']).attrs['sample_rate']

        adjusted_signal = (signal+offset)*(f5range/digitisation)
        current = numpy.array(adjusted_signal, dtype=numpy.float)
        
        # Case 1
        slice_start = int(721376) #int(round(285.03*sampling_rate))
        slice_end = int(739616)#int(round(285.36*sampling_rate))
        # Case 2
#         slice_start = int(1078748)
#         slice_end = int(1098288)

        slice_current = current[slice_start : slice_end]

        # timestep was fADCSequenceInterval * 1e-3 = .01 for .abf
        # Different for .fast5? standard or make use of sampling interval?
        # This is fed into Segmenter as second = 1000/timestep
        timestep = 0.05
        #timestep = 1/sampling_rate
        # sampling_rate = 3012 so would be 0.000332
        # looks weirdly smooth using that, horizontal axis VERY small
        # real value probably somewhere on the order of .01

        # Because each .fast5 file is an event, group events together in 
        # one fileset object that contains all files with same run_id.
        # For new run_id, new fileset, otherwise add to current fileset.

        current = slice_current
        fileset = Fast5FileSet(filename, timestep, current)
        filesets.append(fileset)
        fileset.parse(Segment(current=current, start=0, end=(len(current)-1),
            duration=len(current), second=1000/timestep))

    min_gain_per_sample = 0.05
    sequences = []
    fine_segmentation = None

    # Iterate across filesets and analyze each event by fileset
    for fileset in filesets:
        i = 0
        while i < len(fileset.events):
            event = fileset.events[i]
            filename = fileset.filenames[i]
            timestep = fileset.timesteps[i]
            current = fileset.currents[i]
            second = fileset.seconds[i]

            # Feed event into SpeedyStatSplit, as is done in trnaHMMs.py
            # Alter parameters for best accuracy
            if event.duration > 0:
                event.filter(order=1, cutoff=3000)
                event.parse(SpeedyStatSplit(min_width=minWidth, max_width=maxWidth, \
                                            min_gain_per_sample=min_gain_per_sample, \
                                            window_width=winWidth))
#                 event.parse(StatSplit(min_width=minWidth, max_width=maxWidth, \
#                                             min_gain_per_sample=min_gain_per_sample, \
#                                             window_width=winWidth, splitter='slanted'))

            segment_means = []
            count = 0

            # Create list of segment means
            # (also write means/std to create profiles)
            # REMEMBER TO COMMENT OUT WHEN NOT MAKING NEW PROFILE FILES!
#             print filename
#             new_filename = str(filename.replace('/', '_'))
#             writeFile = open('F5_' + new_filename + str(min_gain_per_sample) + '.txt', 'w')
            for segment in event.segments:
                segment_means.append(segment.mean)
#                 writeFile.write(str(count)+'\t'+str(segment.mean)+'\t'+str(segment.std)+'\n')
                count += 1
#             writeFile.close()

            sequences.append(segment_means)
            sequences = [segment_means]
            
            # Align event to HMM
            pred = prediction(models, sequences, algorithm = 'viterbi')
            scores = [float(pred[0][0])] 
            print event.start, event.end, scores

            classified_model = scores.index(max(scores))
            label = 'P_pro001'
            if classified_model == 0 and label == 'P_pro001':
                p_pro001 += 1
            num_events += 1

            matrix[label][classified_model] += 1

            # plot event according to model using plot_event:
            # top plot is segmented event colored in cycle by segments, bottom 
            # subplot is segmented event aligned with HMM, colored by states
            plot_event(filename, event, model=models[classified_model])

            # iterate through rest of fileset
            i = i + 1

    print >> sys.stderr, '\ntotal time for the program %.3f' % (time.time()-t0)

if (__name__ == '__main__'):
    main()
    raise SystemExit
