#!usr/bin/env python2.7
"""
Program written by Hannah Meyers to analyze nanopore data based on tutorial located:
https://github.com/jmschrei/PyPore/blob/master/examples/PyPore%20Tutorial.ipynb
"""

from __future__ import print_function

from PyPore.parsers import *
from PyPore.DataTypes import *
from hmm import *
from yahmm import *

import PyPore
import math
import matplotlib.pyplot as plt
import itertools as it
import glob
import seaborn as sns
import sys
import pandas as pd
import numpy as np
from proteinDists import *
from scipy.stats import kde



#access data from spreadsheets of mean and std deviation data
#11 columns corresponding to 10 different states of protein data
#two columns for pretitin, usually one is deleted but works much better than just one
profile_means = pd.read_csv( 'profile_data/profilemeans.csv' )
profile_stds = pd.read_csv( 'profile_data/profilestds.csv' )


#Convert CSV data to distribution objects
dists_means = [ NormalDistribution( profile_means[col].mean(), profile_means[col].std() ) for col in profile_means ]
dists_stds = [ LogNormalDistribution( (np.log( profile_stds[col] )).mean(), (np.log( profile_stds[col] )).std() ) for col in profile_stds ]



#build multivariate profile using normal distribution of mean and lognormal distribution of std deviation
profile = [ MultivariateDistribution([ mean, std ]) for mean, std in it.izip( dists_means, dists_stds ) ]

profile

#insert state
insert = MultivariateDistribution( [ UniformDistribution( 0, 40 ), UniformDistribution( 0, 10 ) ] )

#ProteinBoard2 allows for backslips while ProteinDommainBoard does not
#backslips should only occur in beginning between capture and ramping
boardlist = [ProteinDomainBoard2]*2 + [ProteinDomainBoard]*9
#boardlist[4] = ProteinDomainBoard4
#boardlist[3] = ProteinDomainBoard4

#build model
model = ModularDomainProfileModel2( boardlist, profile, "ClpXProfile-{}".format( len(profile) ), insert)

#files to pull experiment and events from for training
filenames = [
#"ProteinFiles/12907001-s05.abf"
#"ProteinFiles/13311001-s05.abf"
"experiment_data/13n25010-s05.abf", #train
#"experiment_data/13n25001-s05.abf",
#"experiment_data/13n25005-s05.abf",
#"experiment_data/13n25007-s05.abf",
#"experiment_data/13n25012-s05.abf", #no full event
#----#
#"experiment_data/13n12001-s05.abf", #train
#"experiment_data/13n13006-s05.abf",
#"experiment_data/14131001-s05.abf",
#---#
#"experiment_data/14410016-s05.abf" #test
]

threshold = 38
rules = [lambda event: event.duration > 1000000,
         lambda event: event.min > -5,
         lambda event: event.max < threshold]


exp = Experiment( filenames )
#parse experiment files
exp.parse( event_detector = lambda_event_parser(threshold = threshold, rules = rules), segmenter = SpeedyStatSplit(min_width = 5, cutoff_freq = 2000., false_positive_rate = 1e-90), filter_params = (1,2000), meta = False)

#access list of events from parsed experiments
events = reduce( list.__add__, [ [ [ ( seg.mean, seg.std ) for seg in event.segments ] for event in file.events ] for file in exp.files ] )

#events for training the model
training_events = events


print("Training Events: {}".format( len(training_events)))

#new model
model.train(training_events, use_pseudocount = True)


#files for testing
filenames = [

#"ProteinFiles/12907001-s05.abf"
#"ProteinFiles/13311001-s05.abf"
"experiment_data/13n25010-s05.abf",
#"experiment_data/13n25001-s05.abf",
#"experiment_data/13n25005-s05.abf",
#"experiment_data/13n25007-s05.abf",
#"experiment_data/13n25012-s05.abf", #no full event
#----#
#"experiment_data/13n12001-s05.abf",
#"experiment_data/13n13006-s05.abf",
#"experiment_data/14131001-s05.abf",
#---#
#"experiment_data/14410016-s05.abf" #long ramping short domains
]

#visualize events with trained model
for file in it.imap( File, filenames ):
    x = 1
    
    print(file.filename)
    #Events must drop below this threshold
    threshold = 38
    rules = [lambda event: event.duration > 1000000,
             lambda event: event.min > -5,
             lambda event: event.max < threshold]
    
    file.parse( lambda_event_parser( threshold=threshold, rules = rules ) )
    
    print(file.events)
    
    for event in file.events:
    
        event.filter()
        
        print(event)
        #false_positive_rate controls the number of segments that will be created by the segmenter
        event.parse( SpeedyStatSplit( min_width=5, false_positive_rate=1e-65, cutoff_freq = 2000) )
        
        #print(event.segments)
        
        #Apply HMM to event
        _, hidden_states = model.viterbi( np.array( [ [ seg.mean, seg.std ] for seg in event.segments ] ) )
        if hidden_states != None:
            
            #First subplot is event + segmentation
            plt.figure( figsize=(20, 8))
            plt.subplot( 311 )
            event.plot( color='cycle' )

            #Second subplot is event + HMM
            plt.subplot( 312 )
            event.plot( color='hmm', hmm=model, hidden_states=hidden_states, cmap='Set1' )

            #Final subplot is color cycle with profile means
            #this subplot is currently inaccurate as it only plots the first profile
            #furthermore, there was a bug in PyPore when I started on this that makes the color cycle
            #not match up to the HMM colors. I am unsure if the bug has been fixed since then.
            ax = plt.subplot( 313 )
            plt.imshow( [ np.arange( 0., len(profile) ) / len(profile) ], interpolation='nearest', cmap="Set1" )
            plt.grid( False )
            means = [ d.parameters[0][0].parameters[0] for d in profile ]
            for i, mean in enumerate( means ):
                plt.text( i-0.2, 0.1, str( round(mean, 1) ), fontsize=12 )
                
            #Output HMM state path to output.txt file
            outputtext = 'output' + str(x) + '.txt'
            f = open(outputtext, 'w')
            for i, state in enumerate( hidden_states ):
                f.write(state[1].name+"\n")
            f.close()
            
        #name png with counter
        #s = file.filename[16:] + 'trained' + str(x)
        s = 'sanitycheck2' + str(x)
        plt.savefig(s)
        x += 1
        #plt.show()
    file.close()




