import os
import json
import h5py
import random
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Generate train/valid JSON for semi-supervised CSV 2026 challenge")

    parser.add_argument("--root", type=str, default="./data", help="Dataset root path")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--val_size", type=int, default=50, help="Number of labeled samples to reserve for validation (will be balanced 1:1 between classes)")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    dataset_root_path = args.root

    images_dir_path = os.path.join(dataset_root_path, 'train', 'images')
    labels_dir_path = os.path.join(dataset_root_path, 'train', 'labels')

    # collect image/label filenames in train
    all_image_filenames = [name for name in os.listdir(images_dir_path) if name.endswith('.h5')]
    all_labeled_filenames = [name.replace('_label', '') for name in os.listdir(labels_dir_path) if name.endswith('.h5')]
    all_unlabeled_filenames = [name for name in all_image_filenames if name not in all_labeled_filenames]

    random.seed(args.seed)

    train_filename_list = [name for name in all_labeled_filenames]

    # build labeled dataset entries and group by class (cls)
    train_labeled_dataset_list = []
    class0_list = []
    class1_list = []
    for label_filenames in train_filename_list:
        image_h5_file_path = os.path.abspath(os.path.join(images_dir_path, label_filenames))
        label_h5_file_path = os.path.abspath(os.path.join(labels_dir_path, label_filenames.replace('.h5', '_label.h5')))
        entry = {
            'image': image_h5_file_path,
            'label': label_h5_file_path
        }
        # try to read class label from the label h5 file
        try:
            with h5py.File(label_h5_file_path, 'r') as hf:
                cls_raw = hf['cls'][()]
                try:
                    cls_val = int(cls_raw)
                except Exception:
                    # fallback for array-like contents
                    if hasattr(cls_raw, "tolist"):
                        cls_val = int(cls_raw.tolist()[0])
                    else:
                        cls_val = int(cls_raw[0])
        except Exception:
            # if reading fails, default to class 0 (shouldn't happen for labeled data)
            cls_val = 0

        train_labeled_dataset_list.append(entry)
        if cls_val == 0:
            class0_list.append(entry)
        else:
            class1_list.append(entry)

    # training set with unlabeled
    train_unlabeled_dataset_list = []
    for label_filenames in all_unlabeled_filenames:
        image_h5_file_path = os.path.abspath(os.path.join(images_dir_path, label_filenames))
        train_unlabeled_dataset_list.append({
            'image': image_h5_file_path,
            'label': None
        })

    # create validation set by sampling from labeled data (balanced by cls)
    val_size = args.val_size
    # ensure even val_size for 1:1 balance
    per_class = val_size // 2
    if per_class == 0:
        valid_dataset_list = []
    else:
        # limit per_class to available samples in each class
        avail0 = len(class0_list)
        avail1 = len(class1_list)
        per_class = min(per_class, avail0, avail1)
        # final val_size may be smaller if not enough samples
        final_val_size = per_class * 2
        valid_dataset_list = []
        if per_class > 0:
            sampled0 = random.sample(class0_list, per_class)
            sampled1 = random.sample(class1_list, per_class)
            valid_dataset_list = sampled0 + sampled1
            # remove sampled entries from train_labeled_dataset_list
            sampled_set = set([e['image'] for e in valid_dataset_list])
            train_labeled_dataset_list = [e for e in train_labeled_dataset_list if e['image'] not in sampled_set]

    # save JSON
    with open(os.path.join(dataset_root_path, 'train_labeled.json'), 'w') as f:
        json.dump(train_labeled_dataset_list, f, indent=4)

    with open(os.path.join(dataset_root_path, 'train_unlabeled.json'), 'w') as f:
        json.dump(train_unlabeled_dataset_list, f, indent=4)

    with open(os.path.join(dataset_root_path, 'valid.json'), 'w') as f:
        json.dump(valid_dataset_list, f, indent=4)

    # print dataset statistics
    # sampled counts (may not exist if no sampling was done)
    sampled0_len = len(sampled0) if 'sampled0' in locals() else 0
    sampled1_len = len(sampled1) if 'sampled1' in locals() else 0

    train_labeled_after_count = len(train_labeled_dataset_list)
    train_unlabeled_count = len(train_unlabeled_dataset_list)

    orig_class0 = len(class0_list)
    orig_class1 = len(class1_list)
    train_class0 = orig_class0 - sampled0_len
    train_class1 = orig_class1 - sampled1_len
    train_labeled_total = train_class0 + train_class1
    val_total = sampled0_len + sampled1_len

    train_class0_pct = (train_class0 / train_labeled_total * 100) if train_labeled_total > 0 else 0.0
    train_class1_pct = (train_class1 / train_labeled_total * 100) if train_labeled_total > 0 else 0.0
    val_class0_pct = (sampled0_len / val_total * 100) if val_total > 0 else 0.0
    val_class1_pct = (sampled1_len / val_total * 100) if val_total > 0 else 0.0

    print("")
    print("=== Dataset split summary ===")
    print("Training set:")
    print(f"  Labeled samples: {train_labeled_after_count}")
    print(f"    - class 0 (low risk): {train_class0} ({train_class0_pct:.1f}%)")
    print(f"    - class 1 (high risk): {train_class1} ({train_class1_pct:.1f}%)")
    print(f"  Unlabeled samples: {train_unlabeled_count}")
    print("")
    print("Validation set:")
    print(f"  Total samples: {val_total}")
    print(f"    - class 0 (low risk): {sampled0_len} ({val_class0_pct:.1f}%)")
    print(f"    - class 1 (high risk): {sampled1_len} ({val_class1_pct:.1f}%)")
    print("=============================")





