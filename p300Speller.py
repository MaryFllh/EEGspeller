
import mne
import numpy as np
import re
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pickle
import scipy.stats
import speech_recognition as sr
import json
#local imports
import priorDistanceBased as prior 
import bvFileReader  
import suggestionWordProbability as swp 
import backEndServer as bs


def handleEndSweep(state, message):
    state['model'] = pickle.load(open(os.path.join(state['path'], state['model_filename']), 'rb')) #load the trained model (LDA model)
    (t0, bvr) = bvFileReader.extractEpoch(state['header_file'], 3, 8, last_t0 = state['t0'], keep_trying = True) #Read the EEG data from the file
    bvr.load_data()
    (events, event_labels) = mne.annotations.events_from_annotations(bvr, bvFileReader.markerForString)   #find the events based on change in the marker value (e.g if the events channel changes from zero an event is detected. We send the stimulus markers in pairs of row/col and zero )
    
    filtered_EEG = bvr.filter(1, 25., fir_design = 'firwin')  #filter the raw EEG with an fir between 1 to 25 Hz
    event_new_dict = {'row/target':4 ,'row/nontarget':5, 'col/target':6, 'col/nontarget':7}   # dictionary of the events. this is dependant what numbers you have assigned to your markers in the design
    baseline = (-.2, 0)   #interval for baseline removal
    sweep_epoch = mne.Epochs(filtered_EEG, events, event_id = event_new_dict, tmin = -.2, tmax = .8,
                baseline = baseline, reject = None, flat = None, preload = True, proj = False, reject_by_annotation = False, verbose = False, on_missing = 'warning')	#epoch the data into windows of -200 ms prior to the stimulus and 800 ms after

    epoch_copy = sweep_epoch.copy()
    #Downsample the epochs
    epoch_downsampled = epoch_copy.resample(50, npad = 'auto') 
    epoch_data = epoch_downsampled.get_data()
    Extracted_data = np.reshape(epoch_data, (len(epoch_downsampled), state['chan_num'] *epoch_data[2].shape[1]))   # concatenate the eeg channels together as features
   
    # classification
    LDA_scores = state['model'].decision_function(Extracted_data)   #spits out scores based on the trained LDA model
    prob = scipy.stats.norm(loc = state['mean'], scale = state['std']).pdf(LDA_scores) #converts that score to a probability based on the (re)trained normal distribution
    prob_distribution = probEachCell(state, layout = state['layout'], probability = prob) #Computes and returns the probability of each cell. at the end of each sweep we have 12 flashes and each flash illuminates 6 cells. so I go in each flash and update the prob of 6 cells that illuminated in that flash.
    priors = prior.gridProbabilities(stub = state['stub'], grid_layout = state['layout'], word_probabilities = state['word_prob'], corpus = state['study_corpus']) #computes the prior probabilties of each cell based on a language model and the string typed up to that point 
    bayes_nominator = np.multiply(priors, prob_dist)
    bayes_denominator = sum(bayes_nominator)
    bayesian_scores = bayes_nominator / bayes_denominator

    if max(bayesian_scores) >= state['threshold'] or state['target_flash_num'] >= state['max_sweeps_online_blocks'] :  
        """state['target_flash_num'] >= M if hasn't reached threshold flashes M times (2M times per cell) 
	(the number is equal to sweeps you want before classifying)
        if we have either passed the thresh or swept state['max_sweep-online_blocks']
        times, then it's time to classify."""
        result = np.argmax(bayesian_scores) + 1	
    else : result = 0    #starts over and flashes again. recomputes everything to increase its confidence about it detection of the user's intention				

    outMessage = bs.Message(bs.MessageType.RESULT, result = result)

	return (state, outMessage)	

def main():

	mr = bs.MessageReceiver()	
	mr.state['header_file'] = header_file_path
	mr.register(bs.MessageType.PRE_SWEEP, handlePreSweep)
	mr.register(bs.MessageType.CONNECTION, handleConnect)
	mr.register(bs.MessageType.BEGIN_SWEEP, handleBeginSweep)
	mr.register(bs.MessageType.FLASH, handleFlash)
	mr.register(bs.MessageType.END_SWEEP, handleEndSweep)  #this function is for classification 
	mr.register(bs.MessageType.FINISHED, handleFinished)
	mr.state = initializeState(mr.state)

	mr.go() # will return here when you set state['should_quit'] = true
		
	
if __name__ == "__main__":
	main()
	
