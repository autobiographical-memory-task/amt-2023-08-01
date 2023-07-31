# AMT 5 Way Scorer

These are the categories that the model identifies:

- specific 
- extended 
- categoric 
- omission 
- associate

## Requirements

The following are required for you project to run.
Don't worry, we'll step through what you need to do to ensure this project will run for you.

- python==3.9.17
- torch==1.13.1
- pytorch-lightning==1.9.0
- numpy==1.24.2
- pandas==1.5.3
- transformers==4.26.0

If you intend on using Python for other projects, it's a good idea to create virtual environments for each of them.
This is because each of your projects will have different dependencies.
These dependencies may interfere with other project dependencies.

### Creating a Virtual Environment

You might want to install Miniconda from the [official website](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#id2).

Once you have installed Miniconda you are ready to use conda on the Anaconda Prompt (Windows) or Terminal (Mac).
At the prompt create a new environment by typing the following below.

    conda create --name amt python==3.9
    conda activate amt

To pull this code into a directory called \<dir\>. This command will create the `<dir>`. The placeholder `<loc>` is the location of this github repository: `https://github.com/undergoer/amt.git`

    git clone <loc> <dir>

## Installation

Navigate to your `<dir>` and ensure that your conda environment is activated as above

    cd <dir>
    pip install -r requirements

## Obtaining the model

The model is very large and needs to be stored in Dropbox. Access the model via this link below and save it in your project `<dir>`.

    https://www.dropbox.com/sh/shad73x1pdqxr9x/AADRii-zJy2DRia9uvzQLRnFa?dl=0

## Usage

On the command line analyse any `<csv-file>` in the following way. The placeholde `<csv-file>` is the name of the csv file you want to score.

    python analyse-csv.py <csv-file>

## Format of the csv-file

Ensure there is one column called 'respones'. If you do not have such a column then you need to specify which column (`<column-name>`) you want scored with the `-r` argument as follows.

    python analyse-csv.py <csv-file> -r <column-name>

