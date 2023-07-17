# Vietnam Landmark Retrieval
This is the repo for simple Vietnam landmark retrieval using Finetuned Resnet152 and local feature matching

## Reformat (test) data directory:
### Database folder
from:
./Landmark_Retrieval/Landmark_Retrieval/test/database/

to:
./Landmark_Retrieval/Landmark_Retrieval/test/database_root/database/
### Query folder
from:
./Landmark_Retrieval/Landmark_Retrieval/test/query_root/query/

to:
./Landmark_Retrieval/Landmark_Retrieval/test/query/

## Note for running
If the csv files don't exist, running the script will create database.csv and query.csv that contain image file name and image label columns. If more data is added (either to database or query), one of 2 should be deleted for a re-run to label the image in database/query directory

Put the model in the same folder as main.py

## Data
https://drive.google.com/file/d/1nQIfFMeQuq2rmYHcYAZGMqpROP4s8kAO/view

## Material (model and video)
https://drive.google.com/drive/folders/1PRkp86G4bno7_jEGi3mYvqKrjpX2M6AE?usp=share_link
