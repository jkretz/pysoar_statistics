
## Installation (tested on macOS)

1. Install Miniconda or Anaconda (if not already done). Follow the [respective instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) for your operating system.

2. Download a zipped version of the repository (Click on Code and you will find a button to download it) and unzip it. 

3. Alternatively, it also can be downloaded via git.
    ```bash
   git clone https://github.com/jkretz/pysoar_statistics.git
    ```
4.  Open a Terminal / Anaconda Prompt (under Windows) and navigate to the (unziped) directory of the repository.

5. Create the conda environment and install the needed anaconda packages:
   ```bash
   conda env create -f environment.yml
   conda activate pysoar_statistics
   ```
   
## Settings 

For each pilot, a directory has to be set up in the `plots` directory. There is an example directory, which contains
the `comp_selection.json`. You need to provide information on the competitions that should be analyzed. 
Here is an example of such an entry:
```
  [
   "2024",
   "dm-zwickau-2024",
   "standard",
   "JK"
  ],
```
The first entry is the year the competition took place. Second and third entry are the competition and the competition
class that should be analyzed. This has to follow the directory structure in the `bin` directory of `pySoar`. 
Last entry is the pilots competition number.

Furthermore, you should set the `ipath_data` to your the `bin` directory of `pySoar`.
## Run

From the `pysoar_statistics` directory, run:

   ```bash
   python pysoar_statistics.py
   ```

This should create the output for the selected pilot.
   
