#Protein Distributions
from yahmm import *
from PyPore.hmm import *

'''
Board functions for protein modeling contained in hmm.py.

ProteinDomainBoard and ProteinDomainBoard2 have very similar transition
probabilities but ProteinDomainBoard2 allows for backslips.

Curretly being used with ModularDomainProfileModel2 where a list of board
functions is passed into the models along with the list of profile means/std
of the 11 different protein domains being modeled. ProteinDomainBoard2 is used 
for the first two 'domains', capture and ramping, as there is some back and forth
between those two states. All subsequent domains are modeled by ProteinDomainBoard
but additional boards will soon be added for better modeling of the specific domains.
'''


def ProteinDomainBoard( distribution, name, insert=UniformDistribution( 0, 90 ) ):
    """
    The current board being used to model each state of the protein HMM.
    
    This board is very simplistic and only models insertions, matches, and 
    deletions with modifications planned to allow for modeling backslips.
    
    The idea with this is to build a base board type which generally models
    protein data, and then make modified versions to model each unique region
    of the trace. This is intended for use with the ModularDomainProfileModel
    which allows multiple profiles and board functions to be passed to it.
    Author: jakob.houser@gmail.com
    """
    
    board = HMMBoard(n=2, name=str(name))
    board.directions = ['>', '>']

    delete = State( None, name="D:{}".format( name ) )
    match = State( distribution, name="M:{}".format( name ) )
    insert = State( insert, name="I:{}".format( name ) )

    #transitions between states
    board.add_transition( board.s1, delete, 1.00 )
    board.add_transition( board.s2, match, 1.00 )

    #deletion transitions
    board.add_transition( delete, board.e1, 0.001 ) #delete to next delete
    board.add_transition( delete, insert, 0.001 )
    board.add_transition( delete, board.e2, 0.998 ) #delete to next match
    
    #insert transitions
    board.add_transition( insert, match, 0.40 )
    board.add_transition( insert, insert, 0.25 )
    board.add_transition( insert, board.e1, 0.001 ) #insert to next delete
    board.add_transition( insert, board.e2, 0.349 ) #insert to next match

    #match transitions
    board.add_transition( match, insert, 0.01 )
    board.add_transition( match, board.e1, 0.01 ) #match to next delete
    board.add_transition( match, board.e2, 0.80 )  #match to next match
    board.add_transition( match, match, 0.18 ) #match loop
    
    return board


def ProteinDomainBoard2( distribution, name, insert=UniformDistribution( 0, 90 ) ): #rewrite
    """
    The current board being used to model each state of the protein HMM.
    
    This board is very simplistic and only models insertions, matches, and 
    deletions with modifications planned to allow for modeling backslips.
    
    The idea with this is to build a base board type which generally models
    protein data, and then make modified versions to model each unique region
    of the trace. This is intended for use with the ModularDomainProfileModel
    which allows multiple profiles and board functions to be passed to it.
    Author: jakob.houser@gmail.com edited by Hannah Meyers
    """
    
    board = HMMBoard(n=3, name=str(name))
    board.directions = ['>', '>', '<']

    delete = State( None, name="D:{}".format( name ) )
    match = State( distribution, name="M:{}".format( name ) )
    insert = State( insert, name="I:{}".format( name ) )
    backslip = State( None, name = "B:{}".format(name))


    #add transitions between these states
    board.add_transition( board.s1, delete, 1.00 )
    board.add_transition( board.s2, match, 1.00 )
    board.add_transition( board.e3, backslip, 1.00)

    #add backslip transitions
    board.add_transition( backslip, match, 0.85) #backslip to prev
    board.add_transition( backslip, board.s3, 0.15)
    
    #add deletion transitions
    board.add_transition( delete, board.e1, 0.001 )
    board.add_transition( delete, insert, 0.001 )
    board.add_transition( delete, board.e2, 0.998 )

    #insert transitions
    board.add_transition( insert, match, 0.40 )
    board.add_transition( insert, insert, 0.25 )
    board.add_transition( insert, board.e1, 0.001 ) #insert to next delete
    board.add_transition( insert, board.e2, 0.349 ) #insert to next match

    #match transitions
    board.add_transition( match, insert, 0.01 )
    board.add_transition( match, match, 0.17 )
    board.add_transition( match, board.e1, 0.01 )   #match to next delete
    board.add_transition( match, board.e2, 0.80 )   #match to next match
    board.add_transition( match, board.s3, 0.01)    #match to backslip
    
    return board

def ProteinDomainBoard4( distribution, name, insert=UniformDistribution( 0, 90 ) ): #rewrite
    """
    Not currently used, but can be used as a template for more specific boards, doesnt currently
    support backslip
    """
    
    board = HMMBoard(n=2, name=str(name))
    board.directions = ['>', '>']

    delete = State( None, name="D:{}".format( name ) )
    match = State( distribution, name="M:{}".format( name ) )
    insert = State( insert, name="I:{}".format( name ) )


    board.add_transition( board.s1, delete, 1.00 )
    board.add_transition( board.s2, match, 1.00 )

    #delete transisitions
    board.add_transition( delete, board.e1, 0.001 )
    board.add_transition( delete, insert, 0.001 )
    board.add_transition( delete, board.e2, 0.998 )
    
    #insert transitions
    board.add_transition( insert, match, 0.40 )
    board.add_transition( insert, insert, 0.25 )
    #board.add_transition( insert, board.e1, 0.001 )
    board.add_transition( insert, board.e2, 0.349 )

    #match transitions
    board.add_transition( match, insert, 0.0199 )
    #board.add_transition( match, board.e1, 0 ) #match to next delete
    board.add_transition( match, board.e2, 0.97 )  #match to next match
    board.add_transition( match, match, 0.1 ) #match loop
    
    return board
