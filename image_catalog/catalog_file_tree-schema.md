## Catalog File Tree Schema

Created by: Jacob A Rose  
Created on: Monday Oct 25th, 2022  

root_dir: `/media/data_cifs/projects/prj_fossils/users/jacob/data/leavesdb-v1_1`

---------
---------
---------
### Part I: Primary datasets
#### Individual Datasets belonging to the main dataset names
(e.g. `Extant_Leaves`, `General_Fossil`, `Fossil`, `Florissant_Fossil`, `PNAS`)

---------
---------
##### 1. Base dataset variant
* Structure:  
├── Extant_Leaves_1024  
│   ├── CSVDataset-config.yaml  
│   ├── Extant_Leaves_1024-full_dataset.csv  
│   └── ImageFileDataset-config.yaml  


* Description:  
	* ***Extant_Leaves_1024*** (variant subdir) (Extant Leaves images smart-resized to `1024x1024` pixels)  
		* ***F1. "CSVDataset-config.yaml"***  
│   ├──── Yaml config describing a CSV catalog.  
│   ├──── Defines the following set of parameters:  
│   ├────── `full_name` (e.g. `Extant_Leaves_1024`): str -> identifier of a dataset + resolution,  
│   ├────── `data_path` : str -> pointing to a csv catalog file,  
│   ├────── `label_encoder_path` : Optional[str] -> pointing to a pickled sklearn LabelEncoder if it exists,  
│   ├────── `subset_key` (e.g. `all`, `train`, `val`, `test`) : str -> Key indicating subset of dataset represented in csv catalog.  
		* ***F2. "Extant_Leaves_1024-full_dataset.csv"***  
│   ├──── CSV catalog containing N rows by M columns, where:  
│   ├────── N = the number of samples in the full dataset, and  
│   ├────── M columns should include {`path`, `family`, `genus`, `species`, `collection`, `catalog_number`}  
		* ***F3. "ImageFileDataset-config.yaml"***  
│   ├──── Yaml config describing the key parameters necessary to extract the CSV dataset from a structured image directory.  
│   ├──── Defines the following set of parameters:  
│   ├────── `base_dataset_name` (e.g. `Extant_Leaves`, `PNAS`, `General Fossil`, etc.): str -> Lowest level identifier of the variant's dataset  
│   ├────── `class_type` (e.g. `family`): str -> column upon which any minimum class thresholding is applied  
│   ├────── `threshold` (e.g. `null`, 3, 5, 20, etc.): Optional[int] -> Specify minimum # of samples per class in order to filter out the most rare classes, if necessary.   
│   ├────── `resolution` (e.g. 512, 1024): int -> Images are each preprocessed offline to be smart-resized so that both `h` and `w` = `resolution`  
│   ├────── `version` (e.g. `v1_0`, `v1_1`) : str -> Version tag for these images and metadata.  
│   ├────── `path_schema` (e.g. `"{family}_{genus}_{species}_{collection}_{catalog_number}"`): str -> filename format in which column names contained within curly-braces are replaced with the corresponding sample's values, and by default these are each separated by underscores `"_"`.

---------
---------

##### 2. Base Dataset + Train-Val-Test splits variant
* Structure:  
	* ├── Extant_Leaves_family_100_1024  
		* │   ├── CSVDataset-config.yaml  
		* │   ├── Extant_Leaves_family_100_1024-full_dataset.csv  
		* │   ├── ImageFileDataset-config.yaml  
		* │   ├── splits  
			* |  ├── splits=(0.5,0.2,0.3)  
				*  │     ├──── family-encoder.pkl  
				*  │     ├──── test_metadata.csv  
				*  │     ├──── train_metadata.csv  
				*  │     ├──── val_metadata.csv  

* Description:  
	* ***Extant_Leaves_family_100_1024*** (Extant Leaves images smart-resized to `1024x1024` pixels, excluding all samples belonging to families with `< 100` images total)  
		* F1. CSVDataset-config.yaml -> (see 1.)  
		* F2. Extant_Leaves_family_100_1024-full_dataset.csv -> (see 1.)  
		* F3. ImageFileDataset-config.yaml -> (see 1.)  
		* ***D1. "splits"***  
			│   ├── subdir containing 1 directory for each unique set of train, val and test splits.  
			
			* ***D1S1. "splits=(0.5,0.2,0.3)"***  
				│   ├── subsubdir containing 3 csv catalog files for train, val, and test splits, plus 1 pickle file containing an encoded `sklearn` `LabelEncoder`.  
				* ***D1S1F1. "{class_type}-encoder.pkl"*** (e.g. `family-encoder.pkl`)  
					│   ├── Pickled class str->int mappings in the form of an `sklearn.preprocessing.LabelEncoder`.  
				* ***D1S1F{2,3,4}. "{train,val,test}_metadata.csv"***  
					│   ├──── CSV catalog containing N_subset rows by M_subset columns, where:  
					│   ├────── N_test = the number of samples in the test  
					│   ├────── M_test columns should include {` ` (a headless index column for recalling original order prior to shuffle & split, `path`  (str path to image on disk), `y` {class_type} preprocessed into encoded integer labels, `family`, `genus`, `species`, `collection`, and `catalog_number` (a unique, externally referentiable identifier)}  

---------
---------  
---------
### Part II: Composite datasets
#### Datasets produced by using set operations to combine dataset catalogs



├── Extant_Leaves_family_100_1024_minus_PNAS_family_100_1024  
│   ├── A_in_B-CSVDataset-config.yaml  
│   ├── B_in_A-CSVDataset-config.yaml  
│   ├── CSVDataset-config.yaml  
│   ├── Extant_Leaves_family_100_1024_in_PNAS_family_100_1024.csv  
│   ├── Extant_Leaves_family_100_1024_minus_PNAS_family_100_1024-full_dataset.csv  
│   ├── inputs  
│   │   ├── A  
│   │   │   └── CSVDataset-config.yaml  
│   │   └── B  
│   │       └── CSVDataset-config.yaml  
│   └── PNAS_family_100_1024_in_Extant_Leaves_family_100_1024.csv  

---------
---------

├── Extant_Leaves_family_100_1024_w_PNAS_family_100_1024  
│   ├── A_in_B-CSVDataset-config.yaml  
│   ├── B_in_A-CSVDataset-config.yaml  
│   ├── CSVDataset-config.yaml  
│   ├── Extant_Leaves_family_100_1024_in_PNAS_family_100_1024.csv  
│   ├── Extant_Leaves_family_100_1024_w_PNAS_family_100_1024-full_dataset.csv  
│   ├── inputs  
│   │   ├── A  
│   │   │   └── CSVDataset-config.yaml  
│   │   └── B  
│   │       └── CSVDataset-config.yaml  
│   └── PNAS_family_100_1024_in_Extant_Leaves_family_100_1024.csv  

---------
---------
---------




104 directories, 247 files
