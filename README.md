# EEGspeller
A software speller controlled by your brain!
I have only uploaded the machine learning implementation of this project. There are many methods that I have not included in  p300Speller.py
To give better insight on this project, there is a 6 x 6 grid of A-Z and some words as a software keyboard. In order to select any of the letters/words each row and column flash while the user attends to the target flashes, i.e. the letter or word they want to select. As the user focuses on the target flashes there will be a peak in the brain signals approximately around 300 ms after the stimulus (flash) onset known as P300. This can be detected through machine learning. 
There are obviously many parts to this implemtation and various scripts that send messages back and forth between the frontend and backend.
More detail on p300Speller.py: 
1-  handleEndSweep is one of the functions that is called at the backend during online classification. It reads the recorded EEG data from 'bvFileReader' and after filtering, epochs the data which is essentially segmenting the data into smaller windows that features will be extracted from (line 25).
2- Data is downsampled from 1000 Hz to 50 Hz and features are extracted (lines 30-32).
3- After feature extraction, the features are fed into the pre-trained model which returns scores for each of the 12 flashes (all 6 rows and 6 columns) (line 34).
4- Scores are converted to probabilities from a normal distribution formed from the training data (line 35).
5- Now we have to discover what are the probabilies of each of the 36 cells in order to detect the user's intention. We have 12 flashes, each corresponding to 6 cells (Note that with each flash, 6 cells illuminate). Since we have sent markers we know which cells correspond to which 12 flashes and therefore probabilites (line 36).
6- Since this is a speller we can leverage the knowledge of natural language in our classifier and weight the cells accordingly. e.g. if the user has selected 'q' there is a high chance that the next letter is 'u'. Line 37 is where these prior probabilities based on a langauge model is computed. 
7- We use the computations from lunes 35-37 to compute posterior probabilities in a Naive bayes (lines 38- 40).
8- Comparing the max probability among the 36 cells or the number of flashes up to that point with a stopping criterion it either makes a decision(returns the cell with the max probabilty as feedback) or starts over and the GUI flashes again.
