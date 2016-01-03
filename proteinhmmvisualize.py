'''
Author: Hannah Meyers

This file contains the experiment code for attempting to model
protein nanopore traces via HMMs. Please see inline comments
for an explanation of what each piece of the code is doing.
'''
from __future__ import print_function

from PyPore.parsers import *
from PyPore.DataTypes import *
from hmm import *
from yahmm import *

import math
import matplotlib.pyplot as plt
import itertools as it
import glob
import seaborn as sns
import sys
import pandas as pd
from proteinDists import *
from scipy.stats import kde



#Experiment data files. The first set before the break are all experiment files from
#the same day of data collection. Files after the break are each from different days.
filenames = [

#"ProteinFiles/12907001-s05.abf"
#"ProteinFiles/13311001-s05.abf"
"experiment_data/13n25010-s05.abf",
#"experiment_data/13n25001-s05.abf",
#"experiment_data/13n25005-s05.abf",
#"experiment_data/13n25007-s05.abf",
#"experiment_data/13n25012-s05.abf",#bad
#----#
#"experiment_data/13n12001-s05.abf",
#"experiment_data/13n13006-s05.abf",
#"experiment_data/14131001-s05.abf",
#---#
#"experiment_data/14410016-s05.abf"
] 

#Inserts are uniform across the range of current we expect to see in an event
insert1 = MultivariateDistribution( [ UniformDistribution( 0, 40 ), UniformDistribution( 0, 10 ) ] )

#Create first local model
profile_means = pd.read_csv( 'profile_data/profilemeans.csv' )
profile_stds = pd.read_csv( 'profile_data/profilestds.csv' )

#Convert CSV data to distribution objects
dists_means = [ NormalDistribution( profile_means[col].mean(), profile_means[col].std() ) for col in profile_means ] 
dists_stds = [ LogNormalDistribution( np.log( profile_stds[col] ).mean(), np.log( profile_stds[col] ).std() ) for col in profile_stds ]


#build multivariate profile with distributions of means/std deviations
profile = [ MultivariateDistribution([ mean, std ]) for mean, std in it.izip( dists_means, dists_stds ) ]
#profile[5] = MultivariateDistribution([ ExtremeValueDistribution( 20, 10 ), LogNormalDistribution( np.log(4.5), np.log(3.5) ) ])

#print(profile[5])

#list of board functions corresponds to the 11 profile positions
boardlist = [ProteinDomainBoard2]*2 +[ProteinDomainBoard]*9

#build model
model = ModularDomainProfileModel2( boardlist, profile, "ClpXProfile-{}".format( len(profile) ), insert1)



#iteration for applying model to events in filenames list and plotting
for file in it.imap( File, filenames ):
    x = 1
    
    print(file.filename)
    #Events must drop below this threshold
    threshold = 38
    rules = [lambda event: event.duration > 1000000,
             lambda event: event.min > -5,
             lambda event: event.max < threshold]
    
    file.parse( lambda_event_parser( threshold=threshold, rules = rules ) )
    
    for event in file.events:
        event.filter()
        
        print(event)
        #false_positive_rate controls the number of segments that will be created by the segmenter
        event.parse( SpeedyStatSplit( min_width=5, false_positive_rate=1e-65, cutoff_freq = 2000) )
        
        #print(event.segments)
        
        
        #Apply HMM to event
        _, hidden_states = model.viterbi( np.array( [[ seg.mean, seg.std] for seg in event.segments ] ) )
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
    
        #s = file.filename[16:] +'fp55s' + str(x)
        s = 'backslip' + str(x)
        #save figure with name s + counter to prevent name duplications
        plt.savefig(s)
        x += 1
        
        #show figure
        #plt.show()
    file.close()
