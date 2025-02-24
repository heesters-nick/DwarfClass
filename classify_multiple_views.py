import csv
import io
import os
import tkinter as tk
from tkinter import ttk

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
from astropy.visualization import simple_norm


def read_h5(file_path):
    """
    Reads cutout data from an HDF5 file.
    Returns a dictionary of datasets.
    """
    with h5py.File(file_path, 'r') as f:
        cutout_data = {}
        for dataset_key in f:
            data = np.array(f[dataset_key])
            cutout_data[dataset_key] = data
    return cutout_data


class ImageClassificationApp:
    def __init__(self, master, h5_data, legacy_dirs, csv_file, with_morphology=False):
        """
        Initialize the image classification application.

        Args:
            master: Tkinter root or Toplevel
            h5_data: Dictionary containing HDF5 data for the four versions
            legacy_dirs: Dictionary containing paths to directories for JPEG images
            csv_file: Path to the CSV file for saving classifications
        """
        self.master = master
        self.h5_data = h5_data
        self.csv_file = csv_file
        self.with_morphology = with_morphology
        self.current_index = self.get_last_index()
        self.current_morphology = None
        self.current_value = None
        self.current_classification_mode = 'dwarf'  # Track which classification we're doing

        # Define morphology options
        self.morphology_options = [
            'dE',
            'dEN',
            'dIrr',
        ]

        # Make the main window expand
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(2, weight=1)

        # Fix the grid: 2 rows × 3 columns
        self.num_rows = 2
        self.num_cols = 3

        # Spacing between cells
        self.horizontal_spacing = 10
        self.vertical_spacing = 10

        # For classification buttons, etc.
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        window_width = max(800, int(screen_width * 0.9))
        window_height = max(600, int(screen_height * 0.9))
        self.master.geometry(f'{window_width}x{window_height}')
        self.master.update_idletasks()

        # Calculate cell dimensions
        self.calculate_dimensions()

        # Build legacy image mappings
        self.legacy_maps = {}
        for key, directory in legacy_dirs.items():
            self.legacy_maps[key] = self.build_legacy_mapping(directory)

        # The 6 images we display for each object
        self.order = [
            'native',
            'binned_2x2',
            'binned_smoothed',
            'r_band_binned_2x2',
            'legacy_context',
            'legacy_enhanced',
        ]

        self.total_images = len(self.h5_data['native']['known_id'])
        self.update_title()

        # Create canvas to display images
        self.canvas = tk.Canvas(master)
        self.canvas.grid(row=0, column=0, columnspan=3, sticky='nsew')

        # Create classification buttons
        self.create_button_frames()

        # Bind keys
        self.bind_keys()

        # Display the first set of images
        self.display_image()

    def create_button_frames(self):
        """Create frames for classification buttons and comment box"""
        # Main container frame
        self.button_frame = ttk.Frame(self.master)
        self.button_frame.grid(row=1, column=0, columnspan=3, sticky='ew', padx=20)

        # Configure weights for centering
        self.button_frame.columnconfigure(0, weight=1)  # Left padding
        self.button_frame.columnconfigure(1, weight=2)  # Panels
        self.button_frame.columnconfigure(2, weight=0)  # Comment box
        self.button_frame.columnconfigure(3, weight=1)  # Right padding

        # Add status label
        self.status_label = ttk.Label(
            self.button_frame,
            text='Currently classifying: Dwarf status (Press Enter to confirm)',
            anchor='center',
        )
        self.status_label.grid(row=0, column=0, columnspan=4, sticky='ew', pady=(5, 5))

        # Create panels container
        panels_container = ttk.Frame(self.button_frame)
        panels_container.grid(row=1, column=1, sticky='ew')

        # Create comment box frame
        comment_frame = ttk.LabelFrame(self.button_frame, text='Comments')
        comment_frame.grid(row=1, column=2, sticky='nsew', padx=(20, 0))

        # Create and configure the comment box
        self.comment_box = tk.Text(comment_frame, width=30, height=10)
        self.comment_box.pack(fill='both', expand=True, padx=5, pady=5)

        # Bind Tab key to move focus out of comment box
        self.comment_box.bind('<Tab>', self.handle_tab)

        # Bind Enter in comment box
        self.comment_box.bind('<Return>', self.handle_enter)

        # Create primary classification panel
        self.primary_panel = ttk.LabelFrame(panels_container, text='Is this a dwarf galaxy?')
        self.primary_panel.pack(fill='x', pady=(0, 5))

        # Create dwarf classification buttons
        self.create_dwarf_buttons()

        # Create morphology panel if enabled
        if self.with_morphology:
            self.morphology_panel = ttk.LabelFrame(panels_container, text='What is the morphology?')
            self.morphology_panel.pack(fill='x', pady=(5, 0))
            self.create_morphology_buttons()

    def handle_tab(self, event):
        """Handle Tab key in comment box to move focus to main window"""
        self.master.focus_set()
        return 'break'  # Prevent default tab behavior

    def create_dwarf_buttons(self):
        """Create buttons for dwarf classification"""
        primary_buttons_frame = ttk.Frame(self.primary_panel)
        primary_buttons_frame.pack(fill='x', padx=20, pady=10)

        for i in range(3):
            primary_buttons_frame.columnconfigure(i, weight=1)

        button_width = 15

        self.no_dwarf_button = ttk.Button(
            primary_buttons_frame,
            text='No Dwarf (1)',
            command=lambda: self.handle_classification(0),
            width=button_width,
            style='TButton',
        )
        self.no_dwarf_button.grid(row=0, column=0, padx=10)

        self.maybe_dwarf_button = ttk.Button(
            primary_buttons_frame,
            text='Maybe Dwarf (2)',
            command=lambda: self.handle_classification(0.5),
            width=button_width,
            style='TButton',
        )
        self.maybe_dwarf_button.grid(row=0, column=1, padx=10)

        self.dwarf_button = ttk.Button(
            primary_buttons_frame,
            text='Dwarf (3)',
            command=lambda: self.handle_classification(1),
            width=button_width,
            style='TButton',
        )
        self.dwarf_button.grid(row=0, column=2, padx=10)

    def create_morphology_buttons(self):
        """Create buttons for morphology classification"""
        morph_buttons_frame = ttk.Frame(self.morphology_panel)
        morph_buttons_frame.pack(fill='x', padx=20, pady=10)

        for i in range(len(self.morphology_options)):
            morph_buttons_frame.columnconfigure(i, weight=1)

        button_width = 15

        self.morph_buttons = {}
        for i, morph in enumerate(self.morphology_options):
            btn = ttk.Button(
                morph_buttons_frame,
                text=f'{morph} ({i + 1})',
                command=lambda m=morph: self.set_morphology(m),
                width=button_width,
                style='TButton',
            )
            btn.grid(row=0, column=i, padx=10)
            self.morph_buttons[morph] = btn

        if self.with_morphology:
            self.update_panel_states('primary')

    def handle_morphology(self, morphology):
        """Handle morphology classification selection"""
        self.current_morphology = morphology
        self.save_classification()

    def bind_keys(self):
        """Bind keyboard shortcuts"""
        # Bind number keys for classification
        for i in range(1, 4):
            self.master.bind(str(i), self.handle_key_press)

        # Bind Enter key for confirmation to main window
        self.master.bind('<Return>', self.handle_enter)

        # Bind Escape key for resetting classifications
        self.master.bind('<Escape>', self.handle_escape)

        # Bind resize event
        self.master.bind('<Configure>', self.on_resize)

    def handle_key_press(self, event):
        """Handle key press events based on current classification mode"""
        # Ignore key presses if focus is in comment box
        if event.widget == self.comment_box:
            return

        key = event.char

        if self.current_classification_mode == 'dwarf':
            if key == '1':
                self.handle_classification(0)
            elif key == '2':
                self.handle_classification(0.5)
            elif key == '3':
                self.handle_classification(1)
        elif self.current_classification_mode == 'morphology':
            if key == '1':
                self.set_morphology(self.morphology_options[0])
            elif key == '2':
                self.set_morphology(self.morphology_options[1])
            elif key == '3':
                self.set_morphology(self.morphology_options[2])

    def handle_enter(self, event):
        """Handle Enter key press for confirming classifications"""
        # Check if we have the required classifications
        can_save = self.current_value is not None and (
            self.current_value == 0  # No dwarf
            or not self.with_morphology  # Morphology disabled
            or (self.with_morphology and self.current_morphology is not None)  # Has morphology
        )

        if can_save:
            # Save and prevent line break if in comment box
            self.save_classification()
            return 'break' if event.widget == self.comment_box else None
        elif event.widget == self.comment_box:
            # If we can't save yet and we're in the comment box, just prevent the line break
            return 'break'

    def handle_escape(self, event):
        """Handle Escape key press to reset classifications"""
        # Only allow reset if we're actually classifying something
        if self.current_value is not None:
            self.reset_classifications()
            # Ensure focus returns to main window for number key classifications
            self.master.focus_set()
        return 'break'

    def reset_classifications(self):
        """Reset all classifications while preserving comments"""
        # Reset all button styles
        self.no_dwarf_button.configure(style='TButton')
        self.maybe_dwarf_button.configure(style='TButton')
        self.dwarf_button.configure(style='TButton')
        if self.with_morphology:
            for btn in self.morph_buttons.values():
                btn.configure(style='TButton')

        # Reset classification values
        self.current_value = None
        self.current_morphology = None if self.with_morphology else None

        # Return to dwarf classification mode
        self.current_classification_mode = 'dwarf'
        self.update_panel_states('primary')

        # Update status label to reflect reset
        self.status_label.configure(
            text='Classifications reset. Currently classifying: Dwarf status (Press Enter to confirm)'
        )

    def set_morphology(self, morphology):
        """Set the current morphology selection"""
        self.current_morphology = morphology

        # Reset all morphology button styles
        for btn in self.morph_buttons.values():
            btn.configure(style='TButton')

        # Highlight selected button
        self.morph_buttons[morphology].configure(style='Selected.TButton')

    def handle_classification(self, value):
        """Handle dwarf classification"""
        self.current_value = value

        # Reset all dwarf button styles and states
        for btn in [self.no_dwarf_button, self.maybe_dwarf_button, self.dwarf_button]:
            btn.configure(style='TButton')

        # Highlight selected button
        if value == 0:
            self.no_dwarf_button.configure(style='Selected.TButton')
        elif value == 0.5:
            self.maybe_dwarf_button.configure(style='Selected.TButton')
        else:
            self.dwarf_button.configure(style='Selected.TButton')

        if value == 0 or not self.with_morphology:
            self.current_morphology = None
            # If morphology is disabled, don't try to change modes
            if not self.with_morphology:
                return
        else:
            self.current_classification_mode = 'morphology'
            self.update_panel_states('morphology')

    def update_panel_states(self, active_panel):
        """Update the visual states of the panels"""
        if not self.with_morphology:
            return

        if active_panel == 'primary':
            self.current_classification_mode = 'dwarf'
            # Enable primary panel buttons
            for btn in [
                self.no_dwarf_button,
                self.maybe_dwarf_button,
                self.dwarf_button,
            ]:
                btn.state(['!disabled'])
            # Disable morphology panel buttons
            for btn in self.morph_buttons.values():
                btn.state(['disabled'])

            self.primary_panel.configure(style='Active.TLabelframe')
            self.morphology_panel.configure(style='Inactive.TLabelframe')
            self.status_label.configure(
                text='Currently classifying: Dwarf status (Press Enter to confirm)'
            )
        else:
            self.current_classification_mode = 'morphology'
            # Disable primary panel buttons but maintain their style
            for btn in [
                self.no_dwarf_button,
                self.maybe_dwarf_button,
                self.dwarf_button,
            ]:
                btn.state(['disabled'])
            # Enable morphology panel buttons
            for btn in self.morph_buttons.values():
                btn.state(['!disabled'])

            self.primary_panel.configure(style='Inactive.TLabelframe')
            self.morphology_panel.configure(style='Active.TLabelframe')
            self.status_label.configure(
                text='Currently classifying: Morphology (Press Enter to confirm)'
            )

    def update_button_states(self, panel, state):
        """Helper method to update button states within a panel"""
        for child in panel.winfo_children():
            for subchild in child.winfo_children():
                if isinstance(subchild, ttk.Button):
                    subchild.state([state])

    def setup_styles(self):
        """Setup custom styles for the panels and buttons"""
        style = ttk.Style()

        # Active panel style
        style.configure('Active.TLabelframe', background='white')
        style.configure(
            'Active.TLabelframe.Label',
            background='white',
            foreground='black',
            font=('TkDefaultFont', 10, 'bold'),
        )

        # Inactive panel style
        style.configure('Inactive.TLabelframe', background='gray90')
        style.configure('Inactive.TLabelframe.Label', background='gray90', foreground='gray50')

        # Selected button styles - both enabled and disabled states
        style.configure('Selected.TButton', background='lightblue')
        style.configure('Selected.Disabled.TButton', background='lightblue')
        style.map(
            'Selected.TButton',
            background=[('disabled', 'lightblue'), ('active', 'skyblue')],
        )

    def build_legacy_mapping(self, directory):
        """
        Build a mapping of object IDs to file paths for legacy images.
        """
        mapping = {}
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                obj_id = filename.split('_')[0]
                mapping[obj_id] = os.path.join(directory, filename)
        return mapping

    def get_last_index(self):
        """
        Determine the last processed index from the CSV file.
        Creates a new CSV file if it doesn't exist.
        """
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                header = 'known_id,label'
                if self.with_morphology:
                    header += ',morphology'
                header += ',comment'  # Add comment column to header
                f.write(header + '\n')
            return 0

        with open(self.csv_file, 'r') as f:
            lines = f.readlines()
            if len(lines) <= 1:
                return 0
            last_line = lines[-1]
            last_id = last_line.split(',')[0]
            last_index = next(
                (
                    i
                    for i, item in enumerate(self.h5_data['native']['known_id'])
                    if last_id in str(item)
                ),
                -1,
            )
            return last_index + 1

    def calculate_dimensions(self):
        """
        Calculate dimensions ensuring square cutouts and space for buttons.
        """
        window_width = max(800, self.master.winfo_width())
        window_height = max(600, self.master.winfo_height())

        # Reserve more space for classification buttons to avoid overlap
        button_space = 150  # Increased from 50 to give more room for buttons

        # Calculate spacing
        total_horizontal_spacing = self.horizontal_spacing * (self.num_cols - 1)
        total_vertical_spacing = self.vertical_spacing * (self.num_rows - 1)

        # Available space for cutouts after reserving button space
        available_width = window_width - 40 - total_horizontal_spacing
        available_height = window_height - button_space - 40 - total_vertical_spacing

        # Calculate cell size to ensure squares
        width_per_cell = available_width // self.num_cols
        height_per_cell = available_height // self.num_rows
        cell_size = min(width_per_cell, height_per_cell)

        self.cell_width = cell_size
        self.cell_height = cell_size

        # Total composite dimensions
        self.composite_width = (self.cell_width * self.num_cols) + total_horizontal_spacing
        self.composite_height = (self.cell_height * self.num_rows) + total_vertical_spacing

    def on_resize(self, event):
        """
        Handle window resize events
        """
        if event.widget == self.master:
            self.master.after(100, self.handle_resize)

    def handle_resize(self):
        """
        Update display after resize
        """
        self.calculate_dimensions()
        self.display_image()

    def update_title(self):
        """
        Update the window title to show progress
        """
        title = f'Classifying Image {self.current_index + 1}/{self.total_images}'
        self.master.title(title)

    def resize_preserve_aspect(self, pil_img, max_width, max_height):
        original_width, original_height = pil_img.size
        if original_width == 0 or original_height == 0:
            return pil_img  # avoid division-by-zero errors

        # Compute scale to fill at least one dimension of the bounding box
        ratio = min(max_width / original_width, max_height / original_height)

        # Calculate new dimensions
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        # Use LANCZOS for resizing
        return pil_img.resize((new_width, new_height), Image.LANCZOS)  # type: ignore

    def convert_to_pil(self, cutout):
        """Convert data to PIL Image using matplotlib's rendering"""
        # Create figure with no borders/axes
        fig = plt.figure(figsize=(6, 6), frameon=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)

        # Create plot with normalization
        with np.errstate(divide='ignore', invalid='ignore'):
            norm = simple_norm(cutout, 'asinh', percent=95.0)
            ax.imshow(cutout, cmap='gray_r', norm=norm, origin='lower')  # type: ignore

        # Save to memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Convert buffer to PIL Image
        buf.seek(0)
        pil_img = Image.open(buf).convert('L')  # Convert to grayscale
        return pil_img

    def display_image(self):
        """
        Arrange the six cutouts (2 rows x 3 cols) in a single composite image,
        each letterboxed inside a cell of identical size.
        """
        self.canvas.delete('all')

        if self.current_index >= self.total_images:
            self.canvas.create_text(
                self.composite_width // 2,
                self.composite_height // 2,
                text='All images classified!',
                font=('Arial', 24),
            )
            return

        # Create a blank composite image
        composite = Image.new(
            'RGB',
            (self.composite_width, self.composite_height),
            color=(255, 255, 255),  # type: ignore
        )
        draw = ImageDraw.Draw(composite)  # noqa: F841

        # Try to load a TTF font, else fallback
        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
        except:  # noqa: E722
            font = ImageFont.load_default()

        # Which object are we on?
        obj_id = self.h5_data['native']['known_id'][self.current_index].decode('utf-8')

        for i, key in enumerate(self.order):
            row = i // self.num_cols
            col = i % self.num_cols

            # Offsets in the composite
            x_offset = col * (self.cell_width + self.horizontal_spacing)
            y_offset = row * (self.cell_height + self.vertical_spacing)

            # Load the image
            if key in self.h5_data:
                cutout_img = self.h5_data[key]['images'][self.current_index]
                if key == 'r_band_binned_2x2':  # Handle single-band image (2D array)
                    pil_img = self.convert_to_pil(cutout_img)
                else:  # Handle regular 3-band images
                    image_array = np.transpose(cutout_img, (1, 2, 0))
                    image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(image_array)
                    pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)  # type: ignore
            elif key in self.legacy_maps:
                mapping = self.legacy_maps[key]
                if obj_id in mapping:
                    try:
                        pil_img = Image.open(mapping[obj_id]).convert('RGB')
                    except Exception:
                        pil_img = Image.new(
                            'RGB',
                            (self.cell_width, self.cell_height),
                            (200, 200, 200),  # type: ignore
                        )
                        ImageDraw.Draw(pil_img).text(
                            (10, 10), 'Error loading', fill='red', font=font
                        )
                else:
                    pil_img = Image.new(
                        'RGB',
                        (self.cell_width, self.cell_height),
                        (220, 220, 220),  # type: ignore
                    )
                    ImageDraw.Draw(pil_img).text((10, 10), 'Not found', fill='red', font=font)
            else:
                pil_img = Image.new(
                    'RGB',
                    (self.cell_width, self.cell_height),
                    (255, 0, 0),  # type: ignore
                )

            # Resize preserving aspect ratio
            pil_img = self.resize_preserve_aspect(pil_img, self.cell_width, self.cell_height)

            # Create a blank cell and center the resized image
            cell_img = Image.new(
                'RGB',
                (self.cell_width, self.cell_height),
                (255, 255, 255),  # type: ignore
            )
            paste_x = (self.cell_width - pil_img.width) // 2
            paste_y = (self.cell_height - pil_img.height) // 2
            cell_img.paste(pil_img, (paste_x, paste_y))

            # Paste cell into the composite
            composite.paste(cell_img, (x_offset, y_offset))

        # Display in the canvas
        self.photo = ImageTk.PhotoImage(composite)
        # Center horizontally and position vertically with proper spacing

        x_center = (self.master.winfo_width() - self.composite_width) // 2
        y_center = 20  # Fixed padding from top

        self.canvas.create_image(x_center, y_center, anchor=tk.NW, image=self.photo)

    def save_classification(self):
        """Save the classification and prepare for next image"""
        cutout_name = self.h5_data['native']['known_id'][self.current_index].decode('utf-8')

        # Get the comment text
        comment = self.comment_box.get('1.0', tk.END).strip()

        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if self.with_morphology:
                if self.current_value == 0:  # No dwarf
                    writer.writerow([cutout_name, self.current_value, '', comment])
                else:
                    writer.writerow(
                        [
                            cutout_name,
                            self.current_value,
                            self.current_morphology,
                            comment,
                        ]
                    )
            else:
                writer.writerow([cutout_name, self.current_value, comment])

        # Reset button styles before moving to next image
        self.no_dwarf_button.configure(style='TButton')
        self.maybe_dwarf_button.configure(style='TButton')
        self.dwarf_button.configure(style='TButton')
        if self.with_morphology:
            for btn in self.morph_buttons.values():
                btn.configure(style='TButton')

        # Reset for next image
        self.current_value = None
        self.current_morphology = None if self.with_morphology else None
        self.current_classification_mode = 'dwarf'

        # Clear the comment box
        self.comment_box.delete('1.0', tk.END)

        # Ensure focus is on main window for number key classifications
        self.master.focus_set()

        # Move to next image
        self.current_index += 1
        self.update_title()
        self.display_image()

        # Reset panel states and status text
        self.update_panel_states('primary')
        self.status_label.configure(
            text='Currently classifying: Dwarf status (Press Enter to confirm)'
        )


if __name__ == '__main__':
    # Adjust these paths to match your own setup
    h5_data_dir = '/home/nick/astro/dwarf_visualization/cutout_data/training_data'

    # HDF5 files for the four preprocessed versions
    h5_paths = {
        'native': os.path.join(h5_data_dir, 'lsb_gri_prep.h5'),
        'binned_2x2': os.path.join(h5_data_dir, 'lsb_gri_prep_binned2x2.h5'),
        'binned_smoothed': os.path.join(h5_data_dir, 'lsb_gri_prep_binned_smoothed.h5'),
        'r_band_binned_2x2': os.path.join(h5_data_dir, 'lsb_r_binned2x2.h5'),
    }

    # Read all HDF5 files
    h5_data = {}
    for key, path in h5_paths.items():
        h5_data[key] = read_h5(path)

    # Optional: directories for the legacy JPEG images
    legacy_dirs = {
        'legacy_context': os.path.join(h5_data_dir, 'train_cutouts_legacy'),
        'legacy_enhanced': os.path.join(h5_data_dir, 'train_cutouts_legacy_enh'),
    }

    # Output CSV file
    csv_file = 'classification_results.csv'

    # Create and run the application
    root = tk.Tk()
    app = ImageClassificationApp(root, h5_data, legacy_dirs, csv_file, with_morphology=True)
    app.setup_styles()
    root.mainloop()
