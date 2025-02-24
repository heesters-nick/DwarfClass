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

1. Make sure you have all necessary data directories and files. You should have:
    - lsb_gri_prep.h5
    - lsb_gri_prep_binned2x2.h5
    - lsb_gri_prep_binned_smoothed.h5
    - lsb_r_binned2x2.h5
    - train_cutouts_legacy/
    - train_cutouts_legacy_enh/

2. Update data paths:
    - h5_data_dir (should contain all of your files ening in .h5)
    - legacy_data_dir (should contain train_cutouts_legacy and train_cutouts_legacy_enh subdirectories)

2. Start the application

3. Classification Process:
    - Answer whether the image shows a dwarf galaxy (No/Maybe/Yes)
    - If "Maybe" or "Yes", classify the morphology (dE, dEN, dI, dIN)
    - Identify any special features (Globular Clusters (GCs), interacting/tidally disturbed, defaults to "No")
    - Add optional comments in the comment box
    - Finally, confirm classifications with Enter key
4. Navigation:
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


