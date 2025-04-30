import h5py
import os

# Print dataset path
dataset_path = "/home/tiejean/Workspace/TaskAgonosticManipulation/LIBERO/libero/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
print(f"Inspecting dataset at: {dataset_path}")

# Open the HDF5 file and list keys
with h5py.File(dataset_path, "r") as d:
    import pdb; pdb.set_trace()
    print("Keys in dataset:", list(d.keys()))
    print("All demos: ", list(d["data"].keys()))
    print("Data format: ", list(d["data"].keys()))
    d["data"]["demo_0"].keys()