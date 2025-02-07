
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import Lock, Manager

import glob
import os
from abc import abstractmethod
from dataclasses import dataclass
import pickle
import random
import shelve
from typing import Callable, List, Optional
from concurrent.futures import ThreadPoolExecutor

import supervision as sv
from tqdm import tqdm

from autodistill.core import BaseModel
from autodistill.detection import CaptionOntology


@dataclass
class ClassificationBaseModel(BaseModel):
    """
    Use a foundation classification model to auto-label data.
    """

    ontology: CaptionOntology

    @abstractmethod
    def predict(self, input: str) -> sv.Classifications:
        """
        Run inference on the model.
        """
        pass

    def predicting(self, img_path: str, fn_post_process: Callable = None, shelve_file: str = None):
        try:
            detections = self.predict(img_path, fn_post_process=fn_post_process)
            if isinstance(shelve_file, str):
                with self.shelf_lock:
                    with shelve.open(shelve_file) as shelf:
                        shelf[img_path] = detections
        except Exception as e:
            detections = None
            print(f"Error labeling {img_path}: {e}")
        return detections

    def label(
        self,
        input_folder: str,
        extension: str = ".jpg",
        output_folder: str | None = None,
        fn_post_process: Callable = None,
        shelve_file: Optional[str] = None,

    ) -> sv.ClassificationDataset:
        """
        Label a dataset and save it in a classification folder structure.
        """
        input_folder = str(input_folder)
        if output_folder is None:
            output_folder = input_folder + "_labeled"
        else:
            output_folder = str(output_folder)

        os.makedirs(output_folder, exist_ok=True)

        image_paths = sorted(glob.glob(input_folder + "/*" + extension))

        img2results = {}
        if shelve_file is not None:
            with shelve.open(shelve_file) as shelf:
                img2results = dict(shelf)
        else:
            shelf = {}

        image_paths = [p for p in image_paths if p not in img2results]
        # Shuffle the image paths
        random.shuffle(image_paths)
        print(f"Started labeling {len(image_paths)} images")

        desc = "Labeling images"
        max_workers = os.cpu_count()
        self.shelf_lock = Lock()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            p = partial(self.predicting, fn_post_process=fn_post_process, shelve_file=shelve_file)
            list(tqdm(
                pool.map(p, image_paths),
                total=len(image_paths),
                desc=desc,
            ))

        with shelve.open(shelve_file) as shelf:
            detections_map = dict(shelf)

        dataset = sv.ClassificationDataset(
            self.ontology.classes(), list(detections_map.keys()), detections_map
        )

        train_cs, test_cs = dataset.split(
            split_ratio=0.7, random_state=None, shuffle=True
        )
        test_cs, valid_cs = test_cs.split(
            split_ratio=0.5, random_state=None, shuffle=True
        )

        train_cs.as_folder_structure(root_directory_path=output_folder + "/train")

        test_cs.as_folder_structure(root_directory_path=output_folder + "/test")

        valid_cs.as_folder_structure(root_directory_path=output_folder + "/valid")

        print("Labeled dataset created - ready for distillation.")
        return dataset
