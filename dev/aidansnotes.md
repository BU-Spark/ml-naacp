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

interesting side effect is that it ends up pulling author names out of the text and using those as tags. see: 
'Charlie Gasparino', 0.5579344034194946), ('Brynn Gingras', 0.557830274105072), ('Napolitano', 0.5545047521591187), ('Brianna Hayward', 0.5496129989624023), ('GUADALAJARA', 0.5461217164993286), ('Ryan Nobles', 0.5397570729255676), ('Sauce', 0.5327435731887817), ('Brenton Tarrant', 0.5267106294631958), ('FoxCast', 0.5218717455863953), ('Adam Klotz', 0.5212829113006592)])

for a blurb on presidents. 


doc2vec infer vector has less than desirable performance on topic inference alone

topicModel handles everything that is needed from the deployment side