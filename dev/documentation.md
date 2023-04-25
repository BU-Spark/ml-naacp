## Dependencies
there's a requirements.txt \n
if you have problems with gensim try:
'pip3 install gensim --no-binary :all:'

## Breakdown

main.py is the key runtime
it features two classes, **pipelining**, which handles web input 
and web output, and **livepredictions** which handles loading and running the engine

firebaseconnect.py is the library responsible for reading and writing to the firebase instance. this is boilerplate code which will write to the database with minimal modification.

rss_acq.py is the library responsible for reading from the RSS feed

topicModel.py is the library which handles the topic identification engine in nearly every capacity. it is exposed via **livepredictions** 

topicModel.py makes use of topicNetwork.py which is the pytorch classifier.

