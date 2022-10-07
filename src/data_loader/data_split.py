import glob
import os
import random
import shutil
import time
from pathlib import Path
from typing import List

import git
import numpy as np
from omegaconf import OmegaConf
from rich.progress import track


def get_patient_id(path: str, splitter: str = "_") -> str:
    """同一患者の画像が，testとtrainにまたがってしまうことがあるので，そのためのidを取得
    path: relative path or filename"""
    file_name = os.path.basename(path)
    patient_id = file_name.split(splitter)[0]
    return patient_id


def get_paths_ids(paths: List[str]):
    """idと，その患者から取れた画像のpath一覧を取得
    [{id: [path1, path2, ...]}, {id: ..."""
    paths_ids = {}
    for path in paths:
        patient_id = get_patient_id(path)
        if patient_id in paths_ids:
            paths_ids[patient_id].append(path)
        else:
            paths_ids[patient_id] = [
                path,
            ]
    return paths_ids


def split(ids: List[str], num_test: int, num_val: int):
    """idベースでtrain/val/testに分割"""
    num_all = len(ids)
    shuffled_ids = random.sample(ids, num_all)
    test_ids = shuffled_ids[:num_test]
    val_ids = shuffled_ids[num_test : num_test + num_val]
    train_ids = shuffled_ids[num_test + num_val :]
    return test_ids, val_ids, train_ids


def select_and_move(
    img_paths, select: bool, dist_dir: str, dist_dir_mask: str = None
) -> None:
    """同じ患者の画像が何枚もtestにあるのはダメなので，test, valのときはidにつき一枚しか画像を取得しない"""
    if select:
        rand_img_path = random.sample(img_paths, 1)
        img_paths = rand_img_path

    for img_path in img_paths:
        file_name = os.path.basename(img_path)
        dist_path = os.path.join(dist_dir, file_name)
        shutil.copy(img_path, dist_path)
        if dist_dir_mask != None:
            mask_path = img_path.replace("Images", "Segmentation_0")
            dist_path = dist_path.replace(dist_dir, dist_dir_mask)
            shutil.copy(mask_path, dist_path)


def select_annotated(paths: List[str]) -> List[str]:
    """アノテーションがない画像を paths から 除外する。"""
    annotated_paths = []
    for path in track(paths, description="select_annotated"):
        if exist_aux_img(path):
            annotated_paths.append(path)

    print(f"Removed the non-annotated images ... {len(annotated_paths)}/{len(paths)}")
    return annotated_paths


def exist_aux_img(path, img_types=["Images", "Segmentation_0", "SubImages"]):
    """img_typesに含まれるものが全部存在するか調べる．"""
    exist_all = True
    for img_type in img_types:
        aux_path = path.replace("Images", img_type)
        if not os.path.exists(aux_path):
            exist_all = False
    return exist_all


def select_RL(img_paths):
    img_paths_r = []
    img_paths_l = []
    for img_path in img_paths:
        if "COLOR_R" in img_path:
            img_paths_r.append(img_path)
        if "COLOR_L" in img_path:
            img_paths_l.append(img_path)
    img_path_r = random.sample(img_paths_r, 1)[0]
    img_path_l = random.sample(img_paths_l, 1)[0]
    return img_path_r, img_path_l


def move_tuple(img_path, dist_dir, img_types):
    for img_type in img_types:
        src_path = img_path.replace("/Images/", "/{}/".format(img_type))
        dist_path = os.path.join(
            dist_dir.replace("Images", img_type), os.path.basename(src_path)
        )
        if img_type == "SubImages":
            src_path += "/SubImage.png"
        # print('src: ', src_path)
        # print('dst: ', dist_path)
        shutil.copy(src_path, dist_path)


def split_RL(paths_ids, num_list):
    ids_rl = []
    ids_single = []
    for idx in paths_ids:
        cnt_r = 0
        cnt_l = 0
        for path in paths_ids[idx]:
            if "COLOR_R" in path:
                cnt_r += 1
            if "COLOR_L" in path:
                cnt_l += 1
        if cnt_r >= 1 and cnt_l >= 1:
            ids_rl.append(idx)
        else:
            ids_single.append(idx)
    random.shuffle(ids_rl)
    new_ids_rl = ids_rl[: num_list[0]]
    new_ids = ids_rl[num_list[0] :] + ids_single
    return new_ids_rl, new_ids


def create_datasets_RL(
    root: str,
    paths: List[str],
    nums_list,
    img_types=["Images", "Segmentation_0", "SubImages"],
    diag_type="NPA",
) -> None:
    paths_ids = get_paths_ids(paths)
    testval_ids, train_ids = split_RL(paths_ids, nums_list)

    # make dirs in proceding
    dist_dirs = {}
    for img_type in img_types:
        for phase in ["test", "val", "train"]:
            dist_dir = os.path.join(root, phase, diag_type, img_type)
            if img_type == "Images":
                dist_dirs[phase] = dist_dir
            os.makedirs(dist_dir)
    print(dist_dirs)

    # test val RL aware
    for idx in track(testval_ids, description="Copying test/val set"):
        img_paths = paths_ids[idx]
        img_path_r, img_path_l = select_RL(img_paths)
        img_path1, img_path2 = random.sample([img_path_r, img_path_l], 2)
        move_tuple(img_path1, dist_dirs["test"], img_types=img_types)
        move_tuple(img_path2, dist_dirs["val"], img_types=img_types)

    # train only
    for idx in track(train_ids, description="Copying train set"):
        img_paths = paths_ids[idx]
        for img_path in img_paths:
            move_tuple(img_path, dist_dirs["train"], img_types=img_types)


def create_datasets(
    root: str,
    paths: List[str],
    nums,
    img_types=["Images", "Segmentation_0", "SubImages"],
    diag_type="nonNPA",
    one_img_person=False,
):
    """分割し，実際に画像をtrain/val/testにうつす。
    This will make, e.g, root/train/sub_dir/ and move images there"""
    paths_ids = get_paths_ids(paths)
    ids = list(paths_ids)
    print("All cnt of patients = ", len(ids))
    phase_ids = {}
    phase_ids["test"], phase_ids["val"], phase_ids["train"] = split(
        ids, num_test=nums.test, num_val=nums.val
    )

    # make dirs in proceding
    dist_dirs = {}
    for img_type in img_types:
        for phase in ["test", "val", "train"]:
            dist_dir = os.path.join(root, phase, diag_type, img_type)
            dist_dirs[phase] = dist_dir
            os.makedirs(dist_dir)

    for phase in ["test", "val", "train"]:
        for idx in track(phase_ids[phase], description="Copying {} set".format(phase)):
            img_paths = paths_ids[idx]
            if one_img_person:
                img_paths = random.sample(img_paths, 1)
            for img_path in img_paths:
                move_tuple(img_path, dist_dirs[phase], img_types=img_types)


def log(cfg, src_dir):
    """データセットの分割をlog"""
    # NOTE dataset/に置かないのは，dvcで管理することになった時に変なことになるから
    repo = git.Repo(search_parent_directories=True)
    git_sha = repo.head.object.hexsha
    text_log = "\n{}\n".format(time.strftime("%Y_%m%d_%H%M"))
    text_log += "git SHA = {}\n".format(git_sha[:8])
    for cur_dir, dirs, files in os.walk(cfg.root):
        if "origin" in cur_dir or "past" in cur_dir:
            continue
        if len(dirs) == 0:
            text_log += "{}: {}\n".format(cur_dir, len(files))

    Path(cfg.logdir).mkdir(exist_ok=True)
    with open(os.path.join(src_dir / cfg.logdir, "data_split.log"), "a") as f:
        f.write(text_log)


def main():
    src_dir: Path = Path(__file__).parent.parent
    cfg = OmegaConf.load(src_dir / "config/split.yaml")
    NPA_dir = str((src_dir / cfg.path_patterns.NPA).resolve())
    print(NPA_dir)
    origin_NPA_paths = glob.glob(NPA_dir)
    nonNPA_dir = str((src_dir / cfg.path_patterns.nonNPA).resolve())
    print(nonNPA_dir)
    origin_nonNPA_paths = glob.glob(nonNPA_dir)

    origin_NPA_paths = select_annotated(origin_NPA_paths)

    create_datasets_RL(
        src_dir / cfg.root, origin_NPA_paths, nums_list=[cfg.nums.NPA.testval, -1]
    )
    create_datasets(
        src_dir / cfg.root,
        origin_nonNPA_paths,
        nums=cfg.nums.nonNPA,
        img_types=["Images"],
        one_img_person=True,
    )

    log(cfg, src_dir)


if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    main()
