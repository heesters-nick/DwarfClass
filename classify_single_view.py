import csv
import tkinter as tk
from tkinter import ttk

import h5py
import numpy as np
from PIL import Image, ImageTk


def read_h5(file_path):
    """
    Reads cutout data from HDF5 file
    :param file_path: path to HDF5 file
    :return: cutout data
    """
    with h5py.File(file_path, 'r') as f:
        # Create a dictionary to store the datasets
        cutout_data = {}
        for dataset_key in f:
            data = np.array(f[dataset_key])
            cutout_data[dataset_key] = data
    return cutout_data


class ImageClassificationApp:
    def __init__(self, master, cutout_data, csv_file):
        self.master = master
        self.cutout_data = cutout_data
        self.csv_file = csv_file
        self.current_index = self.get_last_index()

        self.width = 512
        self.height = 512

        # Initialize classification variable
        self.classification_value = tk.DoubleVar()
        self.classification_value.set(0)
        self.total_images = len(self.cutout_data['known_id'])
        self.update_title()

        # Create canvas for displaying images
        self.canvas = tk.Canvas(master, width=self.width, height=self.height)
        self.canvas.grid(row=0, column=0, columnspan=3)

        # Create buttons for classification
        self.no_dwarf_button = ttk.Button(
            master, text='No Dwarf (1)', command=lambda: self.save_classification(0)
        )
        self.no_dwarf_button.grid(row=1, column=0)

        self.maybe_dwarf_button = ttk.Button(
            master,
            text='Maybe Dwarf (2)',
            command=lambda: self.save_classification(0.5),
        )
        self.maybe_dwarf_button.grid(row=1, column=1)

        self.dwarf_button = ttk.Button(
            master, text='Dwarf (3)', command=lambda: self.save_classification(1)
        )
        self.dwarf_button.grid(row=1, column=2)

        # Bind keys for quick classification
        master.bind('1', lambda event: self.save_classification(0))
        master.bind('2', lambda event: self.save_classification(0.5))
        master.bind('3', lambda event: self.save_classification(1))

        # Display the first image
        self.display_image()

    def get_last_index(self):
        """
        Determines the last processed index from the CSV file.
        """
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w') as f:
                f.write('known_id,label\n')
            return 0

        with open(self.csv_file, 'r') as f:
            lines = f.readlines()
            if len(lines) <= 1:
                return 0
            last_line = lines[-1]
            last_index = next(
                i
                for i, item in enumerate(self.cutout_data['known_id'])
                if str(last_line.split(',')[0]) in str(item)
            )
            return last_index + 1

    def update_title(self):
        """
        Updates the title of the window to show the progress (e.g., "Classifying Image 5/20").
        """
        title = f'Classifying Image {self.current_index + 1}/{self.total_images}'
        self.master.title(title)

    def display_image(self):
        """
        Displays the current image on the canvas.
        """
        if self.current_index >= len(self.cutout_data['known_id']):
            self.canvas.create_text(
                self.width // 2,
                self.height // 2,
                text='All images classified!',
                font=('Arial', 24),
            )
            return

        cutout_img = self.cutout_data['images'][self.current_index]

        # Convert from (channel, height, width) to (height, width, channel)
        image = np.transpose(cutout_img, (1, 2, 0))

        # Convert to uint8 (multiply by 255 and clip)
        image = (image * 255).clip(0, 255).astype(np.uint8)
        # to pil image
        pil_img = Image.fromarray(image)
        # fip upside down for correct viewing
        pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)  # type: ignore
        pil_img = pil_img.resize((self.width, self.height))
        self.photo = ImageTk.PhotoImage(pil_img)

        # Display image on canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def save_classification(self, value):
        """
        Saves the classification value to the CSV file and moves to the next image.
        """
        cutout_name = self.cutout_data['known_id'][self.current_index].decode('utf-8')

        with open(self.csv_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([cutout_name, value])

        self.current_index += 1
        self.update_title()
        self.display_image()


if __name__ == '__main__':
    import os

    # Path to the HDF5 file and CSV file
    h5_file_path = (
        '/home/nick/astro/dwarf_visualization/cutout_data/training_data/lsb_gri_prep.h5'
    )
    csv_file = 'classification_results.csv'

    # Read cutout data
    cutout_data = read_h5(h5_file_path)

    # Start the application
    root = tk.Tk()
    root.title('Image Classification Tool')
    app = ImageClassificationApp(root, cutout_data, csv_file)
    root.mainloop()
