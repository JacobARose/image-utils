"""

image-utils/image_catalog/image_datasets.py

Created on: Thursday Oct 14th, 2022
Created by: Jacob A Rose

"""



# img_root_dir = catalog_registry.available_datasets.get("General_Fossil_family_10_512")
img_root_dir = catalog_registry.available_datasets.get("Fossil_family_10_512")


if isinstance(img_root_dir, (str, Path)):
    class_names = sorted(set(os.listdir(img_root_dir)))
    # class_dirs = [os.listdir(Path(img_root_dir, n)) for n in class_names]
    class_dirs = [os.path.join(img_root_dir, n) for n in class_names]
elif isinstance(img_root_dir, list):
    # Assume a list of image root dirs should be encoded using the union of their unique classes.
    class_names = sorted(set(itertools.chain(*[
        os.listdir(root_i) for root_i in img_root_dir
    ])))
    
    class_dirs = []
    for root_i in img_root_dir:
        class_dirs.extend([os.path.join(root_i, n) for n in os.listdir(root_i)])

class_dirs = sorted(class_dirs, key = lambda x: os.path.split(x)[1])
label_str2int = {n: i for i, n in enumerate(class_names)}

print(f"{img_root_dir=}" + "\n" + f"{len(class_names)=}" + "\n" + f"{len(class_dirs)=}")

path_list = [];
label_list = [];
name_list = [];
# label_str2int = {};

for i, d in enumerate(class_dirs):
    n = os.path.split(d)[1]
    
    # i_paths = sorted(os.listdir(class_dirs[i]))
    i_paths = [
        os.path.join(d, fname) 
        for fname in sorted(os.listdir(d))
    ]
    # i_labels = [i]*len(i_paths)
    path_list.extend(i_paths)
    label_list.extend([i]*len(i_paths))
    name_list.extend([n]*len(i_paths))
    # label_str2int[n] = i

    
# name_list = [class_names[l] for l in label_list]
def get_class_counts(name_list):
    class_counts = Counter(name_list)
    tab_labels = [f"{l}: {class_counts[l]}" for l in name_list]

    return class_counts, tab_labels

class_counts, tab_labels = get_class_counts(name_list)
ipyplot.plot_class_tabs(path_list, tab_labels)