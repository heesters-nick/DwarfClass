# DwarfClass
DwarfClass is a specialized tool for generating expert labels of dwarf galaxies. It provides a structured classification interface that can be used to create training datasets for deep learning models.

## Features

- Sequential classification workflow with three main questions:
  1. Dwarf galaxy identification (No/Maybe/Yes)
  2. Morphology classification (for "Maybe" or "Yes" responses)
  3. Special features identification (defaults to "No" for efficiency)

- User-friendly interface features:
  - Automatic panel progression and highlighting
  - Previous panels are disabled to maintain classification flow
  - Reset option using ESC key for correcting mistakes
  - Comment box for additional observations
  - Enter key confirmation at the end of the classifications
  - Randomized image order to prevent bias

- Progress persistence:
  - Classifications are automatically saved
  - Application can be closed and resumed later
  - Tracks completed classifications and continues with remaining images
  - Saves results in CSV format

## Installation

Clone the repository:
```bash
git clone https://github.com/heesters-nick/DwarfClass.git
cd DwarfClass
```

## Usage

1. Download the data from this link: https://drive.google.com/drive/folders/1l0brBmMbJbm_FzBgGHQiFv4j8vBUUTgf?usp=sharing

2. Make sure you have all necessary data directories and files. You should have:
    - lsb_gri_prep.h5
    - lsb_gri_prep_binned2x2.h5
    - lsb_gri_prep_binned_smoothed.h5
    - lsb_r_binned2x2.h5
    - train_cutouts_legacy/
    - train_cutouts_legacy_enh/

3. Update local data paths in the script `classify_multiple_views_random_order.py`:
    - h5_data_dir (should contain all of your files ening in .h5)
    - legacy_data_dir (should contain train_cutouts_legacy and train_cutouts_legacy_enh subdirectories)

4. Start the application:
  - `python classify_multiple_views_random_order.py`

5. Classification Process:
    - Answer whether the image shows a dwarf galaxy (No/Maybe/Yes)
    - If "Maybe" or "Yes", classify the morphology (dE, dEN, dI, dIN)
    - Identify any special features (Globular Clusters (GCs), interacting/tidally disturbed, defaults to "No")
    - Add optional comments in the comment box
    - Finally, confirm classifications with Enter key
    
6. Navigation:
    - Press Enter to confirm and move to next image
    - If you have made a mistake: press ESC to reset current classification
    - Type in comment box and press Enter to confirm if all other classifications are complete
    - Press Tab to get out of comment box if current classification is incomplete

## Data Management

- Classifications are saved in `classification_results.csv` in the main directory
- Important: Keep this file in the main directory until the classification run is complete
- After completing a run:
    1.  Move the file to your classifications directory
    2. Rename it to `classification_results_[INITIALS]_[RUN NUMBER].csv`
        - Example: `classification_results_NH_1.csv`

## Key Features

- **Randomized Image Order**: Images are presented in random order each time the application starts to prevent classification bias
- **Progress Tracking**: The application remembers which objects have been classified
- **Efficient Workflow**: Default "No" for special features allows quick progression through typical cases
- **Flexible Input**: Complete control over classification process with ability to correct mistakes

## Morphology examples from MATLAS

- Remarks:
  - There are only very few examples for nucleated dwarf irregulars (dIN) in MATLAS
  - Some morphologies may seem different in MATLAS and UNIONS due to varying observing conditions
  - In general: if the galaxy appears smooth -> dE, if there are any features/star forming regions -> dI
  - The colors in the Legacy survey images should not be relied upon for morphological classification; all dwarfs look blue in these images
  

![DwarfClass Interface](images/morph_examples.png)
