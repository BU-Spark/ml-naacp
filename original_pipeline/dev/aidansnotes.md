topicModel handles everything that is needed from the deployment side

topictags, sans .xlsx extension, was a set of human generated tags that are redundant now

tagconfig.json is a polished set of tags extracted from 4 different datasets
it converts tags found in the extraneous datasets (referenced in writeup) to a standard set

when creating the doc2vec encoder, setting the augment flag to true creates another doc2vec sample with each entity found in the article as a valid topic tag

tags are described below

spacy entity tag meanings:
PERSON - People, including fictional.
NORP - Nationalities or religious or political groups.
FAC - Buildings, airports, highways, bridges, etc.
ORG - Companies, agencies, institutions, etc.
GPE - Countries, cities, states.
LOC - Non-GPE locations, mountain ranges, bodies of water.
PRODUCT - Objects, vehicles, foods, etc. (Not services.)
EVENT - Named hurricanes, battles, wars, sports events, etc.
WORK_OF_ART - Titles of books, songs, etc.
LAW - Named documents made into laws.
LANGUAGE - Any named language.
DATE - Absolute or relative dates or periods.
TIME - Times smaller than a day.
PERCENT - Percentage, including "%".
MONEY - Monetary values, including unit.
QUANTITY - Measurements, as of weight or distance.
ORDINAL - "first", "second", etc.
CARDINAL - Numerals that do not fall under another type.

this augmentation has proved to produce much, much higher quality results than any method discussed during the previous semester.  

interesting side effect is that it ends up pulling author names out of the text and using those as tags. 

see:

('Charlie Gasparino', 0.5579344034194946), 
('Brynn Gingras', 0.557830274105072), 
('Napolitano', 0.5545047521591187), 
('Brianna Hayward', 0.5496129989624023), 
('GUADALAJARA', 0.5461217164993286), 
('Ryan Nobles', 0.5397570729255676), 
('Sauce', 0.5327435731887817), 
('Brenton Tarrant', 0.5267106294631958), 
('FoxCast', 0.5218717455863953), 
('Adam Klotz', 0.5212829113006592)])

from passing a nonsensical blurb on presidents as input.


doc2vec infer vector has less than desirable performance on topic inference alone

the way the model is working right now is: 

input text. 
pull out named entities. 
add each named entity as a label. 
pass the same article x times with x labels into the doc2vec model.
train doc2vec model.

use original human edited taggings as supervised labels.
vectorize text using augmented doc2vec.
get vector & similar augmented labels.
pass vector plus manually added label to neural network.
get supervised label output.

as an output we have:

inferred similar augmented entity labels
classifier supervised label

decided to do this twice, so two different engines using different data

given that this is a model trained on gigabytes of news cycle text data 
there's obviously a ton of sensitive language in there. whoever picks up the project 
should be aware and cautious about those coming through as outputs. 

Example below, content warning: 

This is from a CNN article present in the corpus about a post 9/11 hate crime.
 
Entities:  ['9/11', 'Sikh', '110th Street', 'Central Park', 'Sikhs', 'Singh', 'Osama', 'Singh', 'Indian', 'Mesa', 'Arizona', 'Osama bin Laden', 'Muslims', '9/11', .....]

doc2vec unfortunately is going to end up encoding these entities together. 
as the bias is present in the journalism it will be present in the model as well.

in the future if performance is underwhelming on real-world examples i'd strongly encourage
the introduction of another corpus into the training rather than a complete retraining or alteration of the architecture, or a limitation of tags by including them in the droptags.json

to increase classifier performance rather than relevance performance i would drop tags AFTER they are included in doc2vec, ie, drop samples going into the neural network

taglist.json is keyed by tags with which to substitute the list of actual tags

the an instance of topicModel() exposes the tags it's working with via obj.tag_list

in terms of 70 class classification, on open domain text, the doc2vec and neural network perform decently well. the larger doc2vec model is excellent at retrieving relevant entities for the most part and while testing this I often ended up googling tags and learning something new about the subject matter of the article it was processing.

you may find that it is underwhelming on GBH specific data, and I'd assume that this may be due to differences in articles due to authorship and style.

performance can be increased by simply adding more specific tags/editing available tags










