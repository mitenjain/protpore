#!/usr/bin/env python
# Contact: Hannah Meyers
#          hlmeyers@ucsc.edu
# hmm.py

'''
This module serves as a collection for two of the HMMs built for modeling nanopore protein data.
HMMs must be written using Yet Another Hidden Markov Model (yahmm) package. When adding a new
HMM to the list, please follow the pattern set up, and also remember to update the factory at
the bottom. Use the format that the string is the same as the name of the function which returns
that HMM. 

ModularDomainProfileModel was written by Jakob Houser based on example models provided by 
Jacob Schrieber in hmm.py from PyPore

ModularDomainProfileModel2 is a rewrite of ModularDomainProfileModel that takes in
the two lists of profiles and board functions that correspond to the protein domain that
needs to be modeled.

The boards are access
'''

import numpy as np
from yahmm import *

class HMMBoard( Model ):
	"""
	A HMM circuit board. Define the number of lanes in the circuit, with each
	lane having a start and end silent state.
	"""

	def __init__( self, n, name=None ):
		super( HMMBoard, self ).__init__( name="Board {}".format( name ) )

		self.directions = [ '>' ] * n 
		self.n = n
		for i in xrange( 1,n+1 ):
			start = State( None, name="b{}s{}".format(name, i) )
			end = State( None, name="b{}e{}".format(name, i) )

			setattr( self, "s{}".format( i ), start )
			setattr( self, "e{}".format( i ), end )

			self.add_state( start )
			self.add_state( end )


def ModularDomainProfileModel( board_funcs, profiles, name, insert ):
    '''
    Mostly similar to ModuleLocalProfileModel. This model does not support
    insert boards between regions but instead requires a list of board functions
    corresponding to the list of profiles. The two lists should be the same length
    so that board_funcs[i] is the board function to use for the profile at
    profiles[i].
    
    This model is intended to be used with protein modeling where each board_func
    and profile corresponds to a protein domain that needs to be modeled. Critically,
    it is not possible to backslip between domains, though there are no restrictions
    on what transitions are allowed between states within a board_func.
    Author: jakob.houser@gmail.com
    '''
    
    # Initialize the model, and the list of boards in the model
    finalModel = Model( name )
    finalModel.add_transition(finalModel.start, finalModel.end, 1.00)


    # For each profile in the list, add it to the model
    for q, profile in enumerate( profiles ):
        print('q = {}'.format(q))
    
        #board_funcs[q] should correspond to profiles[q]
        board_function = board_funcs[q]
        boards = []
        model = Model( name+":"+str(q) )
        for i, distribution in enumerate( profile ):
            # If this is not the first distribution, then pull the last board to connect to
            if i > 0:
                last_board = boards[-1]

            # If the current distribution is a distribution and not a dictionary..
            if isinstance( distribution, Distribution ):
                # Build a board for that distribution and add it to the model
                board = board_function( distribution, name=i, insert=insert )
                model.add_model( board )

                # If this is the first board, there are no boards to connect to
                if i == 0:
                    boards.append( board )
                    continue

                # If the last board is a single distribution, simply connect to it
                if isinstance( profile[i-1], Distribution ):
                    # Add the current board to the list of boards
                    boards.append( board )

                    # Iterate across all the ports on the board
                    for j, d in it.izip( xrange( 1,board.n+1 ), board.directions ):
                        # Get the end port from the last board and the start port from this board
                        end = getattr( last_board, 'e{}'.format( j ) )
                        start = getattr( board, 's{}'.format( j ) )

                        # Depending on the direction of that port, connect it in the appropriate
                        # direction.
                        if d == '>':
                            model.add_transition( end, start, 1.00 )
                        elif d == '<':
                            model.add_transition( start, end, 1.00 )

                # If the last distribution was actually a dictionary, then we're remerging from a fork.
                elif isinstance( profile[i-1], dict ):
                    # Calculate the number of forks in there
                    n = len( profile[i-1].keys() )

                    # Go through each of the previous boards
                    for last_board in boards[-n:]:
                        for j, d in it.izip( xrange( 1,board.n+1 ), board.directions ):
                            # Get the appropriate end and start port
                            end = getattr( last_board, 'e{}'.format( j ) )
                            start = getattr( board, 's{}'.format( j ) )

                            # Give appropriate transitions given the direction
                            if d == '>':
                                model.add_transition( end, start, 1.00 )
                            elif d == '<':
                                model.add_transition( start, end, 1.00 / n )

                    # Add the board to the growing list
                    boards.append( board )

            # If we're currently in a fork..
            elif isinstance( distribution, dict ):
                # Calculate the number of paths in this fork
                n = len( distribution.keys() )

                # For each path in the fork, attach the boards appropriately
                for key, dist in distribution.items():
                    board = board_function( dist, "{}:{}".format( key, i+1 ), insert=insert )
                    boards.append( board )
                    model.add_model( board )

                    # If the last position was in a fork as well..
                    if isinstance( profile[i-1], dict ):
                        last_board = boards[-n-1]

                        # Connect the ports appropriately
                        for j, d in it.izip( xrange( 1, board.n+1 ), board.directions ):
                            end = getattr( last_board, 'e{}'.format( j ) )
                            start = getattr( board, 's{}'.format( j ) )

                            if d == '>':
                                model.add_transition( end, start, 1.00 )
                            elif d == '<':
                                model.add_transition( start, end, 1.00 )

                    # If the last position was not in a fork, then we need to fork the
                    # transitions appropriately
                    else:
                        # Go through each of the ports and give appropriate transition
                        # probabilities. 
                        for j, d in it.izip( xrange( 1, board.n+1 ), board.directions ):
                            # Get the start and end states
                            end = getattr( last_board, 'e{}'.format( j ) )
                            start = getattr( board, 's{}'.format( j ) )

                            # Give a transition in the appropriate direction.
                            if d == '>':
                                model.add_transition( end, start, 1.00 / n )
                            elif d == '<':
                                model.add_transition( start, end, 1.00 )
        
        board = boards[0]
        initial_insert = State( insert, name="I:0" )
        model.add_state( initial_insert )

        model.add_transition( initial_insert, initial_insert, 0.70 )
        model.add_transition( initial_insert, board.s1, 0.05 )
        model.add_transition( initial_insert, board.s2, 0.25 )

        model.add_transition( model.start, initial_insert, 0.02 )
        model.add_transition( model.start, board.s1, 0.001 )
        model.add_transition( model.start, board.s2, 0.979 )
        
        #Last board transitions to end of current model
        board = boards[-1]
        end_silent = State( None, "S:"+str(q) )
        end_insert = State( insert, "I:"+str(q)+"-end" )
        model.add_transition( board.e2, end_silent, 1.00 )
        model.add_transition( end_silent, model.end, 0.50 )
        model.add_transition( end_silent, end_insert, 0.50 )
        model.add_transition( end_insert, end_insert, 0.99 )
        model.add_transition( end_insert, model.end, 0.01 )

        insert_board = HMMBoard(n=1, name="Insert_Board")
        insert_board.directions = ['>']
        midpt_insert = State(insert, name="I:"+str(q)+"-midpt")
        insert_board.add_states([midpt_insert])
        insert_board.add_transition(insert_board.s1, midpt_insert, 1.00)
        insert_board.add_transition(midpt_insert, midpt_insert, 0.97)
        insert_board.add_transition(midpt_insert, insert_board.e1, 0.03)

            
        #Network union required to combine models
        finalModel.graph = networkx.union( finalModel.graph, model.graph )
        finalModel.add_transition( finalModel.end, model.start, 1.00)
        finalModel.end = model.end
        
    finalModel.bake()
    return finalModel

def ModularDomainProfileModel2( board_funcs, profiles, name, insert ): #rewrite
    '''
    Model currently being used for protein data.
    
    Mostly similar to ModulDomainProfileModel. This model requires a list of board functions
    corresponding to the list of profiles. The two lists should be the same length
    so that board_funcs[i] is the board function to use for the profile at
    profiles[i].
    
    This model is intended to be used with protein modeling where each board_funcs
    and profile corresponds to a protein domain that needs to be modeled. Backslips are
    allowed if the board_funcs passed in supports it. 
    
    Author: Hannah Meyers hlmeyers@ucsc.edu
    '''
    
    # Initialize the model, and the list of boards in the model
    #finalModel = Model( name )
    #finalModel.add_transition(finalModel.start, finalModel.end, 1.00)

    model = Model( name )
    boards = []
    profilelist = ['capture', 'ramping', 'smt3', 'pretitin1', 'pretitin2', 'titintrans', 'preGFP', 'GFPbeta', 'GFPunfold', 'pres2', 'smt33unfold']

    for i, distribution in enumerate( profiles ):
        
        #print(type(distribution))

        # If this is not the first distribution, then pull the last board to connect to
        if i > 0:
            last_board = boards[-1]

        # If the current distribution is a distribution and not a dictionary..
        if isinstance( distribution, Distribution ):

            # Build a board for that distribution and add it to the model
            board = board_funcs[i]( distribution, name=i, insert=insert )
            model.add_model( board )

            # If this is the first board, there are no boards to connect to
            if i == 0:
                boards.append( board )
                continue

            # If the last board is a single distribution, simply connect to it
            if isinstance( profiles[i-1], Distribution ):
                # Add the current board to the list of boards
                boards.append( board )

                # Iterate across all the ports on the board
                for j, d in it.izip( xrange( 1,board.n+1 ), board.directions ):
                    # Get the end port from the last board and the start port from this board
                    end = getattr( last_board, 'e{}'.format( j ) )
                    start = getattr( board, 's{}'.format( j ) )

                    # Depending on the direction of that port, connect it in the appropriate
                    # direction.
                    if d == '>':
                        model.add_transition( end, start, 1.00 )
                    elif d == '<':
                        model.add_transition( start, end, 1.00 )

            # If the last distribution was actually a dictionary, then we're remerging from a fork.
            elif isinstance( profile[i-1], dict ):
                # Calculate the number of forks in there
                n = len( profile[i-1].keys() )

                # Go through each of the previous boards
                for last_board in boards[-n:]:
                    for j, d in it.izip( xrange( 1,board.n+1 ), board.directions ):
                        # Get the appropriate end and start port
                        end = getattr( last_board, 'e{}'.format( j ) )
                        start = getattr( board, 's{}'.format( j ) )

                        # Give appropriate transitions given the direction
                        if d == '>':
                            model.add_transition( end, start, 1.00 )
                        elif d == '<':
                            model.add_transition( start, end, 1.00 / n )

                # Add the board to the growing list
                boards.append( board )

        # If we're currently in a fork..
        elif isinstance( distribution, dict ):
            # Calculate the number of paths in this fork
            n = len( distribution.keys() )

            # For each path in the fork, attach the boards appropriately
            for key, dist in distribution.items():
                board = board_function( dist, "{}:{}".format( key, i+1 ), insert=insert )
                boards.append( board )
                model.add_model( board )

                # If the last position was in a fork as well..
                if isinstance( profile[i-1], dict ):
                    last_board = boards[-n-1]

                    # Connect the ports appropriately
                    for j, d in it.izip( xrange( 1, board.n+1 ), board.directions ):
                        end = getattr( last_board, 'e{}'.format( j ) )
                        start = getattr( board, 's{}'.format( j ) )

                        if d == '>':
                            model.add_transition( end, start, 1.00 )
                        elif d == '<':
                            model.add_transition( start, end, 1.00 )

                # If the last position was not in a fork, then we need to fork the
                # transitions appropriately
                else:
                    # Go through each of the ports and give appropriate transition
                    # probabilities. 
                    for j, d in it.izip( xrange( 1, board.n+1 ), board.directions ):
                        # Get the start and end states
                        end = getattr( last_board, 'e{}'.format( j ) )
                        start = getattr( board, 's{}'.format( j ) )

                        # Give a transition in the appropriate direction.
                        if d == '>':
                            model.add_transition( end, start, 1.00 / n )
                        elif d == '<':
                            model.add_transition( start, end, 1.00 )
        
        board = boards[0]
        initial_insert = State( insert, name="I:0" )
        #backslip = State(None, name= "B:0")
        model.add_state( initial_insert )

        model.add_transition( initial_insert, initial_insert, 0.70 )
        model.add_transition( initial_insert, board.s1, 0.05 )
        model.add_transition( initial_insert, board.s2, 0.25 )

        #model.add_transition( model.start, initial_insert, 0.02 )
        #model.add_transition( model.start, board.s1, 0.001 )
        #model.add_transition( model.start, board.s2, 0.979 )
        
        #ensure goes to ramping, does not allow deletion or insertion for first state
        model.add_transition( model.start, board.s2, 1 )

        
        #Last board transitions to end of current model
        board = boards[-1]
        end_silent = State( None, "S:"+str(i) )
        end_insert = State( insert, "I:"+str(i)+"-end" )
        model.add_transition( board.e2, end_silent, 1.00 )
        model.add_transition( end_silent, model.end, 0.50 )
        model.add_transition( end_silent, end_insert, 0.50 )
        model.add_transition( end_insert, end_insert, 0.99 )
        model.add_transition( end_insert, model.end, 0.01 )

        
    model.bake()
    return model


hmm_factory = {}
