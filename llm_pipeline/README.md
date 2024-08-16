# Project Pipeline - Local Setup Guide

## Overview

This guide will help you set up and run the pipeline locally. It covers required files, installation of dependencies, and common troubleshooting tips.

## Required Files

Ensure you have the following files and resources:

1. **LLM Model (3.1, 8B Version)**:
   - Download the LLaMA model 3.1, 8B version. Place it in the `models/llama_3_1_8b/` directory.

2. **Location Files**:
   - Necessary for geolocation tasks. Ensure these files are placed in the `data_prod/` directory.
        - known_locations.json
        - unwanted_locations.json

3. **Topic Files**
   - Necessary for topic modeling. Ensure these files are placed in the `data_prod/` directory.
        - Asad_Topics_List.xlsx
        - Content_Taxonomy.csv
        - embedding_similarity_label.csv

4. **Articles to Process**:
   - Store your articles in the `data_set/` directory in csv format with required fields:
        - Headline, Publisher, Byline, Paths, Publish Date, Body, content_id
        - Ensure Date is of the following format: Mon Mar 20 10:07:11 EST 2023

5. **Required Keys**:
   - API keys or access tokens for external services. Store them securely in an environment file (`.env`).

## Installation of Dependencies

To install the necessary dependencies, follow these steps:

1. **Python Environment**:
   - Ensure Python 3.8 or higher is installed on your system.

2. **Install Python Dependencies**:
   - Install the required Python packages using `pip`:

3. **Download NER Model**:
   - Download and install the required NER model:

     ```bash
     python -m spacy download en_core_web_sm
     ```

## Running the Pipeline

Once the dependencies are installed and the required files are in place, you can run the pipeline found here:

- Final_Pipeline.ipynb