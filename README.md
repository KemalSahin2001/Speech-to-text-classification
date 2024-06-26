# Project Overview

This app is built using Streamlit, utilizing a BERT model for classification and Google's Speech-to-Text for audio transcription. The repository is organized for clarity and ease of use, ensuring contributors can navigate and utilize the structure efficiently.

## Directory and File Structure

```
project/
│
├── data/
│   ├── external/                   # Data from third-party sources
│   ├── processed/                  # Cleaned and preprocessed data ready for analysis
│   │   ├── cleaned_overview_of_recordings.csv  # Cleaned recordings overview
│   └── raw/                        # Original, unprocessed datasets
│       ├── recordings/             # Folder containing raw recordings
│       └── overview-of-recordings.csv  # Initial overview of recordings
│
├── models/
│   ├── runs/                       # Model run directories with timestamps
│   ├── wav2vec2-finetuned/         # Fine-tuned Wav2Vec2 model
│   └── bert_model_full.pth         # Full BERT model checkpoint
│
├── notebooks/
│   ├── eda.ipynb                   # Notebook for exploratory data analysis
│   ├── speech-to-text_train-v2.ipynb  # Notebook for speech-to-text training (v2)
│   ├── speech-to-text_train.ipynb  # Notebook for speech-to-text training
│   └── text-classification.ipynb   # Notebook for text classification
│
├── streamlit app/
│   └── app.py                      # Streamlit application script
│
├── Reports/
│   ├── AIN413_KemalSahin_proposal.pdf  # Project proposal document
│   └── MI_in_healthcare_technical_report.pdf  # Technical report document
│
├── README.md                       # Documentation and overview of the project
└── requirements.txt                # Dependencies for the project
```

## Getting Started

To begin working with the project, follow these steps:

1. **Clone the repository to your local machine:**

   ```bash
   git clone https://github.com/KemalSahin2001/Speech-to-text-classification.git
   
   cd Speech-to-text-classification

2. **Create a Miniconda environment using Python 3.11:**

   ```bash
   conda create --name speechEnv python=3.11
   ```

3. **Activate the newly created environment:**

   ```bash
   conda activate speechEnv
   ```

4. **Install 'pip-tools':**

   ```bash
   pip install pip-tools
   ```

5. **Use `requirements.in` file for installing dependencies:**

   ```bash
   pip-compile requirements.in
   ```

6. **Install requirements using pip:**

   ```bash
   pip install -r requirements.txt
   ```

7. **Download the datasets manually from Kaggle and place them in their respective directories if needed:**

   - Medical Speech Transcription and Intent: [Download from Kaggle](https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent)
   
8. **Train the BERT model using the provided notebook (`text-classification.ipynb`), as the trained model is too large to upload to GitHub.**

9. **Start with the `eda.ipynb` to get a walkthrough of the project setup and initial analysis.**

## Project Description

### Data

- **External Data:** Data from third-party sources.
- **Processed Data:** Cleaned and preprocessed data ready for analysis, including `cleaned_overview_of_recordings.csv`.
- **Raw Data:** Original, unprocessed datasets, including recordings and the initial overview file `overview-of-recordings.csv`.

### Models

- **Runs:** Directories containing specific model runs with timestamps.
- **Wav2Vec2:** Fine-tuned Wav2Vec2 model for speech-to-text tasks.
- **BERT Model:** Full BERT model checkpoint for text classification (to be trained using the provided notebook).

### Notebooks

- **EDA:** Exploratory Data Analysis notebook.
- **Speech-to-Text Training:** Notebooks for training speech-to-text models.
- **Text Classification:** Notebook for text classification tasks.

### Source Code

- **Streamlit App:** A script for running a Streamlit application.

### Reports

- **Project Proposal:** AIN413_KemalSahin_proposal.pdf
- **Technical Report:** MI_in_healthcare_technical_report.pdf

## Additional Resources

- **Demo Video:** Watch a demo of the Streamlit app [here](https://www.youtube.com/watch?v=6JO8EN9k9vU).

## Additional Notes

- Ensure you have the necessary permissions and access to the datasets.
- Familiarity with basic terminal operations, Conda, and Python environments is assumed.

> This guide includes environment setup, package installation, and initial steps to begin project analysis with a Jupyter notebook. For detailed instructions on each step, refer to the respective notebooks.
