import os

# Set your folder path here
folder_path = r"\\10.16.59.34\CullenLab_Server\Current Project Databases - NHP\2025 Cerebellum prosthesis\Nike\20251125_NRR_RW010\Video\DLC"

for filename in os.listdir(folder_path):
    if filename.startswith("NRR_RW010__") and ".csv" in filename:
        new_filename = filename.replace("NRR_RW010__", "NRR_RW010_")
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")