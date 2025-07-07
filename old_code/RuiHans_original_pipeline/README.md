# Oculomotor_Pipelines
 Contains the matlab code used to process behavior and neural data from Primate experiment with read & write probe
 behavior pipeline can be used to process behavior data, specifically eye position data 
 Neural pipeline can be used to process neural data recoded by NP10 read&write probe 
 Neural Data Analysis is the code used to analyze the neural data after spike sorting together with the previous behavior data

## Behavior Pipeline

### **artifact_filter.m** 
MATLAB code for 

## Neural Pipeline

### Usage Instructions
1. First run ***get_matching_intan_blackrock.m***. the code will generate a sorted intan folder where each intan file sorted and copied to a folder named after the corresponding blackrock files. 

2. Then run ***All_Intan2Bin.m***
The code will first prompt:

**root folder for sorted intan files:**
> This is the root folder for all the sorted intan files from previous steps. 
then the code would prompt for 

**select folder to store the output:**
> This folder is used to store the output of the code: including **session trigger files** (which is used in later steps), **channel mapping schemetic** (red for stim channels and green for the others), and the ***.bin*** file for all the intan files.
The code will then prompt 

**name session .bin file:**
> This ask for an arbitrary name for the .bin file. The default is **all_files_**.

3. The ***.bin*** file is then used for spike sorting using ***kilosort4***. 

4. The kilosort result needs to be manually curated in phy gui with 
phy template-gui params.py
be sure to mark all the "good" clusters green This is **important** for running the code ***KiloSort2Stitch_KS4_NStruct.m***

5. After manually curating the spike sort result, pass it into ***KiloSort2Stitch_KS4_NStruct.m*** to stitch it back to the ***_neural.mat*** file for further analysis. 
***KiloSort2Stitch_KS4_NStruct.m*** will first prompt 
choose root folder for _neural.mat files to stitch kilosort 4 result
for the folder for the _neural.mat files that is previously generated and calibrated from intan files 

**select folder for session trigger files:**

> this ask for folder for session trigger files generated in step 2

**select folder for kilosort4 results files:**

> this ask for the folder for the kilosort4 results. 

**select output folder:**

> prompt for the folder to store the _neural.mat files with unit activity (ua) and firing rate estimate (fr). 

### Code Specifics
### **get_matching_intan_blackrock.m**
The code match the black rock files to the corresponding intan files based on the last modified dates of the ***setting.xml*** and ***.ccf*** files. Such that we can keep the numbering of the neural files constitent with the blackrock files. 

#### **All_Intan2Bin.m**
Adapt the approach of concatenating all Intan recordings from multiple stimulation trials into a single ***.bin*** file for spike sorting using **Kilosort4**. The default channel map setting, ImecPrimateStimRec128_kilosortChanMap.mat, contains the coordinates (in µm) of the **128** channels on the **Neuropixel read &write probe**. The mapping of Neuropixel channel arrangement to the numbering in Intan files is predefined in the code as ***neuropixel_index***. Additionally, the code utilizes ***artifact_Removal.m*** to eliminate artifacts in the recordings caused by stimulation onset. The parameters for artifact removal can be fine-tuned for each specific session using the scripts in filter test.

### **KiloSort2Stitch_KS4_NStruct.m**
The code stitch the result of spike sorting back to the raw neural data struct. The code match the spike times detected by kilosort4 and the time stamp on the neural data. The process add two new fields (unit activity (ua) and firing rate estimate (fr)) to the data struct. Note that fr is estimated from ua using kaiser filter of at a cutoff frequency tailored to the data and a sampling rate of 1000. 

