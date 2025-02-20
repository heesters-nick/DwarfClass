import tkinter as tk
from tkinter import ttk
import h5py
import numpy as np
import csv
from PIL import Image, ImageTk, ImageDraw
import os
from astropy.visualization import simple_norm
import matplotlib.pyplot as plt
import io
import random


def read_h5(file_path):
    """
    Reads cutout data from an HDF5 file.
    Returns a dictionary of datasets.
    """
    with h5py.File(file_path, "r") as f:
        cutout_data = {}
        for dataset_key in f:
            data = np.array(f[dataset_key])
            cutout_data[dataset_key] = data
    return cutout_data


class ImageClassificationApp:
    def __init__(
        self,
        master,
        h5_data,
        legacy_dirs,
        csv_file,
        with_morphology=False,
        show_object_id=False,
    ):
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
        self.show_object_id = show_object_id

        # Prepopulate CSV if needed
        self.prepopulate_csv_if_needed()

        # Total number of images from the HDF5 file
        self.total_images = len(self.h5_data["native"]["known_id"])

        # Load unclassified indices (rows that need classification) from CSV
        self.unclassified_indices = self.get_unclassified_indices()
        random.shuffle(self.unclassified_indices)
        self.random_index_ptr = 0  # pointer within the randomized list

        self.current_morphology = None
        self.current_value = None
        self.current_classification_mode = (
            "dwarf"  # Track which classification we're doing
        )

        # Define morphology options
        self.morphology_options = [
            "dE",
            "dEN",
            "dIrr",
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
        self.master.geometry(f"{window_width}x{window_height}")
        self.master.update_idletasks()

        # Calculate cell dimensions
        self.calculate_dimensions()

        # Build legacy image mappings
        self.legacy_maps = {}
        for key, directory in legacy_dirs.items():
            self.legacy_maps[key] = self.build_legacy_mapping(directory)

        # The 6 images we display for each object
        self.order = [
            "native",
            "binned_2x2",
            "binned_smoothed",
            "r_band_binned_2x2",
            "legacy_context",
            "legacy_enhanced",
        ]

        self.total_images = len(self.h5_data["native"]["known_id"])
        self.update_title()

        # Create canvas to display images
        self.canvas = tk.Canvas(master)
        self.canvas.grid(row=0, column=0, columnspan=3, sticky="nsew")

        # Create classification buttons
        self.create_button_frames()

        # Bind keys
        self.bind_keys()

        # Display the first set of images
        self.display_image()

    def prepopulate_csv_if_needed(self):
        """
        If the CSV file does not exist, create it and write a header and one row per object ID,
        with blank classification fields.
        """
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["known_id", "label", "morphology", "comment"])
                for obj in self.h5_data["native"]["known_id"]:
                    writer.writerow([obj.decode("utf-8"), "", "", ""])

    def get_unclassified_indices(self):
        """
        Reads the CSV file and returns a list of row indices (0-indexed, corresponding to the
        order in the HDF5 file) that do not yet have a valid classification.
        A classification is valid if:
          - The label is non-empty AND
            - If the label is "0", the morphology field must be blank.
            - If the label is nonzero (e.g. "0.5" or "1"), the morphology field must be filled.
        """
        unclassified = []
        with open(self.csv_file, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
            # Skip header; row i (starting at 0) corresponds to CSV row i+1.
            for i, row in enumerate(rows[1:]):
                label = row[1].strip()
                morphology = row[2].strip() if len(row) > 2 else ""
                if label == "":
                    unclassified.append(i)
                else:
                    if label == "0":
                        if (
                            morphology != ""
                        ):  # Should be blank for a valid classification
                            unclassified.append(i)
                    else:
                        if morphology == "":  # Must be filled if label is nonzero
                            unclassified.append(i)
        return unclassified

    def create_button_frames(self):
        """Create frames for classification buttons and comment box"""
        # Main container frame
        self.button_frame = ttk.Frame(self.master)
        self.button_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=20)

        # Configure weights for centering
        self.button_frame.columnconfigure(0, weight=1)  # Left padding
        self.button_frame.columnconfigure(1, weight=2)  # Panels
        self.button_frame.columnconfigure(2, weight=0)  # Comment box
        self.button_frame.columnconfigure(3, weight=1)  # Right padding

        # Add status label
        self.status_label = ttk.Label(
            self.button_frame,
            text="Currently classifying: Dwarf status (Press Enter to confirm)",
            anchor="center",
        )
        self.status_label.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(5, 5))

        # Create panels container
        panels_container = ttk.Frame(self.button_frame)
        panels_container.grid(row=1, column=1, sticky="ew")

        # Create comment box frame
        comment_frame = ttk.LabelFrame(self.button_frame, text="Comments")
        comment_frame.grid(row=1, column=2, sticky="nsew", padx=(20, 0))

        # Create and configure the comment box
        self.comment_box = tk.Text(comment_frame, width=30, height=10)
        self.comment_box.pack(fill="both", expand=True, padx=5, pady=5)

        # Bind Tab key to move focus out of comment box
        self.comment_box.bind("<Tab>", self.handle_tab)

        # Bind Enter in comment box
        self.comment_box.bind("<Return>", self.handle_enter)

        # Create primary classification panel
        self.primary_panel = ttk.LabelFrame(
            panels_container, text="Is this a dwarf galaxy?"
        )
        self.primary_panel.pack(fill="x", pady=(0, 5))

        # Create dwarf classification buttons
        self.create_dwarf_buttons()

        # Create morphology panel if enabled
        if self.with_morphology:
            self.morphology_panel = ttk.LabelFrame(
                panels_container, text="What is the morphology?"
            )
            self.morphology_panel.pack(fill="x", pady=(5, 0))
            self.create_morphology_buttons()

    def handle_tab(self, event):
        """Handle Tab key in comment box to move focus to main window"""
        self.master.focus_set()
        return "break"  # Prevent default tab behavior

    def create_dwarf_buttons(self):
        """Create buttons for dwarf classification"""
        primary_buttons_frame = ttk.Frame(self.primary_panel)
        primary_buttons_frame.pack(fill="x", padx=20, pady=10)

        for i in range(3):
            primary_buttons_frame.columnconfigure(i, weight=1)

        button_width = 15

        self.no_dwarf_button = ttk.Button(
            primary_buttons_frame,
            text="No Dwarf (1)",
            command=lambda: self.handle_classification(0),
            width=button_width,
            style="TButton",
        )
        self.no_dwarf_button.grid(row=0, column=0, padx=10)

        self.maybe_dwarf_button = ttk.Button(
            primary_buttons_frame,
            text="Maybe Dwarf (2)",
            command=lambda: self.handle_classification(0.5),
            width=button_width,
            style="TButton",
        )
        self.maybe_dwarf_button.grid(row=0, column=1, padx=10)

        self.dwarf_button = ttk.Button(
            primary_buttons_frame,
            text="Dwarf (3)",
            command=lambda: self.handle_classification(1),
            width=button_width,
            style="TButton",
        )
        self.dwarf_button.grid(row=0, column=2, padx=10)

    def create_morphology_buttons(self):
        """Create buttons for morphology classification"""
        morph_buttons_frame = ttk.Frame(self.morphology_panel)
        morph_buttons_frame.pack(fill="x", padx=20, pady=10)

        for i in range(len(self.morphology_options)):
            morph_buttons_frame.columnconfigure(i, weight=1)

        button_width = 15

        self.morph_buttons = {}
        for i, morph in enumerate(self.morphology_options):
            btn = ttk.Button(
                morph_buttons_frame,
                text=f"{morph} ({i + 1})",
                command=lambda m=morph: self.set_morphology(m),
                width=button_width,
                style="TButton",
            )
            btn.grid(row=0, column=i, padx=10)
            self.morph_buttons[morph] = btn

        if self.with_morphology:
            self.update_panel_states("primary")

    def bind_keys(self):
        """Bind keyboard shortcuts"""
        # Bind number keys for classification
        for i in range(1, 4):
            self.master.bind(str(i), self.handle_key_press)

        # Bind Enter key for confirmation to main window
        self.master.bind("<Return>", self.handle_enter)

        # Bind Escape key for resetting classifications
        self.master.bind("<Escape>", self.handle_escape)

        # Bind resize event
        self.master.bind("<Configure>", self.on_resize)

    def handle_key_press(self, event):
        """Handle key press events based on current classification mode"""
        # Ignore key presses if focus is in comment box or if all classifications are done
        if event.widget == self.comment_box or self.random_index_ptr >= len(
            self.unclassified_indices
        ):
            return

        key = event.char

        if self.current_classification_mode == "dwarf":
            if key == "1":
                self.handle_classification(0)
            elif key == "2":
                self.handle_classification(0.5)
            elif key == "3":
                self.handle_classification(1)
        elif self.current_classification_mode == "morphology":
            if key == "1":
                self.set_morphology(self.morphology_options[0])
            elif key == "2":
                self.set_morphology(self.morphology_options[1])
            elif key == "3":
                self.set_morphology(self.morphology_options[2])

    def handle_enter(self, event):
        """Handle Enter key press for confirming classifications"""
        # First check if we're done with all classifications
        if self.random_index_ptr >= len(self.unclassified_indices):
            return "break" if event.widget == self.comment_box else None

        # Check if we have the required classifications
        can_save = self.current_value is not None and (
            self.current_value == 0  # No dwarf
            or not self.with_morphology  # Morphology disabled
            or (
                self.with_morphology and self.current_morphology is not None
            )  # Has morphology
        )

        if can_save:
            # Save and prevent line break if in comment box
            self.save_classification()
            return "break" if event.widget == self.comment_box else None
        elif event.widget == self.comment_box:
            # If we can't save yet and we're in the comment box, just prevent the line break
            return "break"

    def handle_escape(self, event):
        """Handle Escape key press to reset classifications"""
        # Only allow reset if we're actually classifying something
        if self.current_value is not None:
            self.reset_classifications()
            # Ensure focus returns to main window for number key classifications
            self.master.focus_set()
        return "break"

    def reset_classifications(self):
        """Reset all classifications while preserving comments"""
        # Reset all button styles
        self.no_dwarf_button.configure(style="TButton")
        self.maybe_dwarf_button.configure(style="TButton")
        self.dwarf_button.configure(style="TButton")
        if self.with_morphology:
            for btn in self.morph_buttons.values():
                btn.configure(style="TButton")

        # Reset classification values
        self.current_value = None
        self.current_morphology = None if self.with_morphology else None

        # Return to dwarf classification mode
        self.current_classification_mode = "dwarf"
        self.update_panel_states("primary")

        # Update status label to reflect reset
        self.status_label.configure(
            text="Classifications reset. Currently classifying: Dwarf status (Press Enter to confirm)"
        )

    def set_morphology(self, morphology):
        """Set the current morphology selection"""
        self.current_morphology = morphology

        # Reset all morphology button styles
        for btn in self.morph_buttons.values():
            btn.configure(style="TButton")

        # Highlight selected button
        self.morph_buttons[morphology].configure(style="Selected.TButton")

    def handle_classification(self, value):
        """Handle dwarf classification"""
        self.current_value = value

        # Reset all dwarf button styles and states
        for btn in [self.no_dwarf_button, self.maybe_dwarf_button, self.dwarf_button]:
            btn.configure(style="TButton")

        # Highlight selected button
        if value == 0:
            self.no_dwarf_button.configure(style="Selected.TButton")
        elif value == 0.5:
            self.maybe_dwarf_button.configure(style="Selected.TButton")
        else:
            self.dwarf_button.configure(style="Selected.TButton")

        if value == 0 or not self.with_morphology:
            self.current_morphology = None
            # If morphology is disabled, don't try to change modes
            if not self.with_morphology:
                return
        else:
            self.current_classification_mode = "morphology"
            self.update_panel_states("morphology")

    def update_panel_states(self, active_panel):
        """Update the visual states of the panels"""
        if not self.with_morphology:
            return

        if active_panel == "primary":
            self.current_classification_mode = "dwarf"
            # Enable primary panel buttons
            for btn in [
                self.no_dwarf_button,
                self.maybe_dwarf_button,
                self.dwarf_button,
            ]:
                btn.state(["!disabled"])
            # Disable morphology panel buttons
            for btn in self.morph_buttons.values():
                btn.state(["disabled"])

            self.primary_panel.configure(style="Active.TLabelframe")
            self.morphology_panel.configure(style="Inactive.TLabelframe")
            self.status_label.configure(
                text="Currently classifying: Dwarf status (Press Enter to confirm)"
            )
        else:
            self.current_classification_mode = "morphology"
            # Disable primary panel buttons but maintain selected style
            for btn in [
                self.no_dwarf_button,
                self.maybe_dwarf_button,
                self.dwarf_button,
            ]:
                # Don't change the style, just disable the button
                btn.state(["disabled"])

            # Enable morphology panel buttons
            for btn in self.morph_buttons.values():
                btn.state(["!disabled"])

            self.primary_panel.configure(style="Inactive.TLabelframe")
            self.morphology_panel.configure(style="Active.TLabelframe")
            self.status_label.configure(
                text="Currently classifying: Morphology (Press Enter to confirm)"
            )

    def setup_styles(self):
        """Setup custom styles for the panels and buttons"""
        style = ttk.Style()

        # Active panel style with blue border
        style.configure(
            "Active.TLabelframe",
            background="white",
            borderwidth=2,  # Increase border width
            relief="solid",  # Solid border style
        )
        style.configure(
            "Active.TLabelframe.Label",
            background="white",
            foreground="black",
            font=("TkDefaultFont", 10, "bold"),
        )
        # Add blue border around active panel
        style.map(
            "Active.TLabelframe",
            bordercolor=[("!disabled", "#007bff")],  # Bootstrap-like blue
        )

        # Inactive panel style
        style.configure(
            "Inactive.TLabelframe", background="gray90", borderwidth=1, relief="solid"
        )
        style.configure(
            "Inactive.TLabelframe.Label", background="gray90", foreground="gray50"
        )
        style.map("Inactive.TLabelframe", bordercolor=[("!disabled", "gray70")])

        # Disabled panel style
        style.configure(
            "Disabled.TLabelframe", background="gray80", borderwidth=1, relief="solid"
        )
        style.configure(
            "Disabled.TLabelframe.Label", background="gray80", foreground="gray60"
        )
        style.map("Disabled.TLabelframe", bordercolor=[("!disabled", "gray60")])

        # Default button style - grey appearance
        style.configure("TButton", background="gray90")

        # Selected button style - blue appearance
        style.configure("Selected.TButton", background="lightblue")
        style.map(
            "Selected.TButton",
            background=[("active", "skyblue"), ("disabled", "lightblue")],
            relief=[("disabled", "sunken")],
            foreground=[("disabled", "gray50")],
        )

    def build_legacy_mapping(self, directory):
        """
        Build a mapping of object IDs to file paths for legacy images.
        """
        mapping = {}
        for filename in os.listdir(directory):
            if filename.lower().endswith((".jpg", ".jpeg")):
                obj_id = filename.split("_")[0]
                mapping[obj_id] = os.path.join(directory, filename)
        return mapping

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
        self.composite_width = (
            self.cell_width * self.num_cols
        ) + total_horizontal_spacing
        self.composite_height = (
            self.cell_height * self.num_rows
        ) + total_vertical_spacing

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

    def count_valid_classifications(self):
        valid = 0
        with open(self.csv_file, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                label = row[1].strip()
                morphology = row[2].strip() if len(row) > 2 else ""
                # A valid classification has a non-empty label and (if label is nonzero) a non-empty morphology.
                if label and (
                    (label == "0" and morphology == "") or (label != "0" and morphology)
                ):
                    valid += 1
        return valid

    def update_title(self):
        """Updates the window title with progress and current object ID"""
        valid_count = self.count_valid_classifications()

        if self.random_index_ptr >= len(self.unclassified_indices):
            self.master.title("All images classified!")
            return

        # Get current object ID from the randomized list
        current_obj_index = self.unclassified_indices[self.random_index_ptr]
        current_obj_id = self.h5_data["native"]["known_id"][current_obj_index].decode(
            "utf-8"
        )

        if self.show_object_id:
            self.master.title(
                f"Classifying {current_obj_id} ({valid_count + 1}/{self.total_images})"
            )
        else:
            self.master.title(
                f"Classifying object {valid_count + 1}/{self.total_images}"
            )

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
        return pil_img.resize((new_width, new_height), Image.LANCZOS)

    def convert_to_pil(self, cutout):
        """Convert data to PIL Image using matplotlib's rendering"""
        # Create figure with no borders/axes
        fig = plt.figure(figsize=(6, 6), frameon=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)

        # Create plot with normalization
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = simple_norm(cutout, "asinh", percent=95.0)
            ax.imshow(cutout, cmap="gray_r", norm=norm, origin="lower")

        # Save to memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # Convert buffer to PIL Image
        buf.seek(0)
        pil_img = Image.open(buf).convert("L")  # Convert to grayscale
        return pil_img

    def display_image(self):
        """
        Displays the image cutouts for the current object using the index from the randomized
        list of unclassified indices.
        """

        self.canvas.delete("all")

        # Check completion state first
        is_complete = self.random_index_ptr >= len(self.unclassified_indices)

        # If complete, disable everything before updating display
        if is_complete:
            # Disable all controls first
            self.disable_all_controls()

            # Create a celebratory completion image
            try:
                # Load and display the completion image
                completion_img = Image.open(".done.jpg")

                # Calculate dimensions to fit in canvas while preserving aspect ratio
                canvas_width = self.master.winfo_width()
                canvas_height = self.composite_height

                # Resize image to fit canvas while maintaining aspect ratio
                img_width, img_height = completion_img.size
                scale = min(canvas_width / img_width, canvas_height / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)

                completion_img = completion_img.resize(
                    (new_width, new_height), Image.LANCZOS
                )
                self.completion_photo = ImageTk.PhotoImage(completion_img)

                # Center the image on canvas
                x_center = canvas_width // 2
                y_center = canvas_height // 2

                self.canvas.create_image(
                    x_center, y_center, image=self.completion_photo, anchor=tk.CENTER
                )

                # Add completion text below the image
                self.canvas.create_text(
                    x_center,
                    y_center + (new_height // 2) + 20,
                    text="All images classified!",
                    font=("Arial", 24),
                    fill="black",
                )
            except FileNotFoundError:
                # Fallback to text only if image not found
                self.canvas.create_text(
                    self.composite_width // 2,
                    self.composite_height // 2,
                    text="All images classified!",
                    font=("Arial", 24),
                )
            return

        # Rest of the existing display_image code remains the same...
        current_obj_index = self.unclassified_indices[self.random_index_ptr]
        obj_id = self.h5_data["native"]["known_id"][current_obj_index].decode("utf-8")

        # Create a blank composite image to serve as a canvas for the grid
        composite = Image.new(
            "RGB", (self.composite_width, self.composite_height), color=(255, 255, 255)
        )

        for i, key in enumerate(self.order):
            row = i // self.num_cols
            col = i % self.num_cols
            x_offset = col * (self.cell_width + self.horizontal_spacing)
            y_offset = row * (self.cell_height + self.vertical_spacing)

            # Load image from h5 or legacy mapping as before...
            if key in self.h5_data:
                cutout_img = self.h5_data[key]["images"][current_obj_index]
                if key == "r_band_binned_2x2":
                    pil_img = self.convert_to_pil(cutout_img)
                else:
                    image_array = np.transpose(cutout_img, (1, 2, 0))
                    image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(image_array)
                    pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
            elif key in self.legacy_maps:
                mapping = self.legacy_maps[key]
                if obj_id in mapping:
                    try:
                        pil_img = Image.open(mapping[obj_id]).convert("RGB")
                    except Exception:
                        pil_img = Image.new(
                            "RGB", (self.cell_width, self.cell_height), (200, 200, 200)
                        )
                        ImageDraw.Draw(pil_img).text(
                            (10, 10), "Error loading", fill="red"
                        )
                else:
                    pil_img = Image.new(
                        "RGB", (self.cell_width, self.cell_height), (220, 220, 220)
                    )
                    ImageDraw.Draw(pil_img).text((10, 10), "Not found", fill="red")
            else:
                pil_img = Image.new(
                    "RGB", (self.cell_width, self.cell_height), (255, 0, 0)
                )

            pil_img = self.resize_preserve_aspect(
                pil_img, self.cell_width, self.cell_height
            )
            cell_img = Image.new(
                "RGB", (self.cell_width, self.cell_height), (255, 255, 255)
            )
            paste_x = (self.cell_width - pil_img.width) // 2
            paste_y = (self.cell_height - pil_img.height) // 2
            cell_img.paste(pil_img, (paste_x, paste_y))
            composite.paste(cell_img, (x_offset, y_offset))

        self.photo = ImageTk.PhotoImage(composite)
        x_center = (self.master.winfo_width() - self.composite_width) // 2
        y_center = 20
        self.canvas.create_image(x_center, y_center, anchor=tk.NW, image=self.photo)

    def save_classification(self):
        """
        Saves the classification for the current object by updating the appropriate row
        in the CSV file rather than appending a new row.
        Then, it advances the randomized pointer so that the same object is not classified twice.
        """
        # Check if we're done with all classifications
        if self.random_index_ptr >= len(self.unclassified_indices):
            return

        # Get the current object index from the randomized list.
        current_obj_index = self.unclassified_indices[self.random_index_ptr]
        obj_id = self.h5_data["native"]["known_id"][current_obj_index].decode("utf-8")
        comment = self.comment_box.get("1.0", tk.END).strip()

        # Determine morphology field: if label is 0 then leave it blank.
        morphology = "" if self.current_value == 0 else self.current_morphology

        # Update the CSV row corresponding to this object.
        self.update_csv_row(
            current_obj_index, obj_id, self.current_value, morphology, comment
        )

        # Reset button styles and fields as before.
        self.no_dwarf_button.configure(style="TButton")
        self.maybe_dwarf_button.configure(style="TButton")
        self.dwarf_button.configure(style="TButton")
        if self.with_morphology:
            for btn in self.morph_buttons.values():
                btn.configure(style="TButton")
        self.current_value = None
        self.current_morphology = None if self.with_morphology else None
        self.current_classification_mode = "dwarf"
        self.comment_box.delete("1.0", tk.END)
        self.master.focus_set()

        # Advance to the next object.
        self.random_index_ptr += 1

        # Check if we're done after advancing
        if self.random_index_ptr >= len(self.unclassified_indices):
            self.disable_all_controls()
            self.update_title()
            self.display_image()
        else:
            self.update_title()
            self.display_image()
            self.update_panel_states("primary")
            self.status_label.configure(
                text="Currently classifying: Dwarf status (Press Enter to confirm)"
            )

    def update_csv_row(self, row_index, known_id, classification, morphology, comment):
        """
        Updates the CSV file in place with additional ID verification.

        Args:
            row_index: Index of the object in the h5 file
            known_id: ID of the object being classified
            classification: Classification value (0, 0.5, or 1)
            morphology: Morphology classification (if applicable)
            comment: User comment

        Raises:
            ValueError: If the row_index doesn't correspond to the expected object ID
        """
        with open(self.csv_file, "r", newline="") as f:
            reader = list(csv.reader(f))

        # Verify the ID at the target row matches our expected ID
        target_row = reader[row_index + 1]  # +1 for header
        existing_id = target_row[0]

        if existing_id != known_id:
            raise ValueError(
                f"ID mismatch at row {row_index + 1}: "
                f"Expected '{known_id}', found '{existing_id}'. "
                "This indicates a potential synchronization issue between the data and CSV."
            )

        # If verification passes, proceed with update
        new_row = [known_id, str(classification), morphology, comment]
        reader[row_index + 1] = new_row

        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(reader)

    def disable_all_controls(self):
        """Disable all buttons, comment box, and key bindings when classification is complete"""
        # Grey out panels first
        self.primary_panel.configure(style="Disabled.TLabelframe")
        if self.with_morphology:
            self.morphology_panel.configure(style="Disabled.TLabelframe")

        # Then disable and style buttons
        for btn in [self.no_dwarf_button, self.maybe_dwarf_button, self.dwarf_button]:
            btn.state(["disabled"])
            if btn["style"] == "Selected.TButton":
                btn.configure(style="Selected.Disabled.TButton")
            else:
                btn.configure(style="Disabled.TButton")
            btn["command"] = ""

        # Disable morphology buttons if present
        if self.with_morphology:
            for btn in self.morph_buttons.values():
                btn.state(["disabled"])
                if btn["style"] == "Selected.TButton":
                    btn.configure(style="Selected.Disabled.TButton")
                else:
                    btn.configure(style="Disabled.TButton")
                btn["command"] = ""

        # Disable comment box and grey it out
        self.comment_box.configure(state="disabled", background="gray80")

        # Unbind all key shortcuts
        self.master.unbind("<Return>")
        for i in range(1, 4):
            self.master.unbind(str(i))
        self.master.unbind("<Escape>")
        self.comment_box.unbind("<Tab>")
        self.comment_box.unbind("<Return>")

        # Disable handlers
        self.handle_classification = lambda x: None
        self.set_morphology = lambda x: None
        self.handle_enter = lambda x: None
        self.handle_escape = lambda x: None
        self.handle_key_press = lambda x: None

        # Update status label
        self.status_label.configure(
            text="All images have been classified!", foreground="gray50"
        )


if __name__ == "__main__":
    # Adjust these paths to match your own setup
    h5_data_dir = "/home/nick/astro/dwarf_visualization/cutout_data/training_data"

    # HDF5 files for the four preprocessed versions
    h5_paths = {
        "native": os.path.join(h5_data_dir, "lsb_gri_prep.h5"),
        "binned_2x2": os.path.join(h5_data_dir, "lsb_gri_prep_binned2x2.h5"),
        "binned_smoothed": os.path.join(h5_data_dir, "lsb_gri_prep_binned_smoothed.h5"),
        "r_band_binned_2x2": os.path.join(h5_data_dir, "lsb_r_binned2x2.h5"),
    }

    # Read all HDF5 files
    h5_data = {}
    for key, path in h5_paths.items():
        h5_data[key] = read_h5(path)

    # Optional: directories for the legacy JPEG images
    legacy_dirs = {
        "legacy_context": os.path.join(h5_data_dir, "train_cutouts_legacy"),
        "legacy_enhanced": os.path.join(h5_data_dir, "train_cutouts_legacy_enh"),
    }

    # Output CSV file
    csv_file = "classification_results.csv"

    # Create and run the application
    root = tk.Tk()
    app = ImageClassificationApp(
        root,
        h5_data,
        legacy_dirs,
        csv_file,
        with_morphology=True,
        show_object_id=True,
    )
    app.setup_styles()
    root.mainloop()
