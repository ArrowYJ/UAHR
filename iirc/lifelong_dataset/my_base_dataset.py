import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
import warnings
from contextlib import contextmanager
from typing import Any, Optional, Callable, List, Tuple, Dict

import iirc.utils.utils

from iirc.definitions import NO_LABEL_PLACEHOLDER, CIL_SETUP, IIRC_SETUP, DatasetStructType


class BaseDataset(ABC):

    def __init__(self,
                 dataset: DatasetStructType,
                 tasks: List[List[str]],
                 setup: str = IIRC_SETUP,
                 using_image_path: bool = False,
                 cache_images: bool = False,
                 essential_transforms_fn: Optional[Callable[[Any], Any]] = None,
                 augmentation_transforms_fn: Optional[Callable[[Any], Any]] = None,
                 test_mode: bool = False,
                 complete_information_mode: Optional[bool] = None,
                 superclass_data_pct: float = 0.6,
                 subclass_data_pct: float = 0.6,
                 superclass_sampling_size_cap: int = 100):
        self.essential_transforms_fn = essential_transforms_fn
        self.using_image_path = using_image_path
        self.dataset = dataset
        self.tasks = tasks
        self.total_dataset_size = len(dataset)
        self.test_mode = test_mode
        setups = (CIL_SETUP, IIRC_SETUP)
        assert setup in setups, f"invalid setup {setup}, choices are {CIL_SETUP} and {IIRC_SETUP}"
        self.setup = setup

        if augmentation_transforms_fn is not None:
            self._apply_augmentations = True
            self.augmentation_transforms_fn = augmentation_transforms_fn
        else:
            self._apply_augmentations = False

        self.n_tasks = len(self.tasks)

        if self.setup == IIRC_SETUP:
            if complete_information_mode is not None:
                self.complete_information_mode = complete_information_mode
            else:
                self.complete_information_mode = self.test_mode

        if self.setup == IIRC_SETUP and not self.test_mode:
            self.superclass_data_pct = superclass_data_pct
            self.subclass_data_pct = subclass_data_pct
            self.superclass_sampling_size_cap = superclass_sampling_size_cap
            self.task_id_to_data_idx = self._divide_data_across_tasks_IIRC(superclass_data_pct, subclass_data_pct,
                                                                           superclass_sampling_size_cap)
        else:
            self.task_id_to_data_idx = self._divide_data_across_tasks()

        # Task specific properties, need to be updated in increment_task and choose_task functions
        self.cur_task = tasks[0]
        self.cur_task_id = 0
        self.cur_task_data_ids = []
        self._load_cur_task_data()
        self.seen_classes = []
        self._update_seen_classes()

        # used only if using_image_path is True (self.dataset includes the images paths and not the PIL images)
        self.cache_images = cache_images
        self.cached_images = {}

        # attributes not to be saved and loaded
        self.non_savable_attributes = ["dataset", "essential_transforms_fn", "augmentation_transforms_fn"]

    def dataset_state_dict(self) -> Dict:
        """
        This function returns a dict that contains the current state of the dataset

        Returns:
            Dict: a dictionary with all the attributes (key is attribute name) and their values, except the
            attributes in the self.non_savable_attributes
        """
        state_dict = {}
        for key in self.__dict__.keys():
            if key not in self.non_savable_attributes:
                state_dict[key] = self.__dict__[key]
        return state_dict

    def load_dataset_state_dict(self, state_dict: Dict) -> None:
        """
        This function loads the object attributes with the values in state_dict

        Args:
            state_dict (Dict): a dictionary with the attribute names as keys and their values
        """
        for key in state_dict.keys():
            self.__dict__[key] = state_dict[key]

    def reset(self) -> None:
        """
        Reset the dataset to the starting state
        """
        self.cur_task = self.tasks[0]
        self.cur_task_id = 0
        self.cur_task_data_ids = []
        self._load_cur_task_data()
        self.seen_classes = []
        self._update_seen_classes()

    def _divide_data_across_tasks_IIRC(self, superclass_data_pct, subclass_data_pct,
                                       superclass_sampling_size_cap) -> Dict[int, List[int]]:

        dataset = self.dataset
        tasks = self.tasks
        cla_to_data_idx = {}
        labels = [sample[1] for sample in dataset]
        classes_combinations = list(set(labels))
        # cap the superclass_data_pct to superclasses with more than 8 subclasses
        superclass_2_n_subclasses = {}
        for classes_combination in classes_combinations:
            length_cls_comb=len(classes_combination)
            if length_cls_comb >= 2:
                superclass = classes_combination[-2]
                if superclass not in superclass_2_n_subclasses.keys():
                    superclass_2_n_subclasses[superclass] = 1
                else:
                    superclass_2_n_subclasses[superclass] += 1

                if length_cls_comb == 3:
                    rootclass=classes_combination[0]
                    ################### root classes
                    if rootclass not in superclass_2_n_subclasses.keys():
                        superclass_2_n_subclasses[rootclass] = 8

        superclass_2_data_pct = {}
        for superclass in superclass_2_n_subclasses:
            n_subclasses = superclass_2_n_subclasses[superclass]
            n_subclasses_cap = superclass_sampling_size_cap
            if n_subclasses > n_subclasses_cap:
                superclass_2_data_pct[superclass] = n_subclasses_cap / n_subclasses * superclass_data_pct
            else:
                superclass_2_data_pct[superclass] = superclass_data_pct

        ########### split data

        for classes_combination in classes_combinations:
            data_subset_idx = [sample_id for sample_id in range(len(dataset))
                               if dataset[sample_id][1] == classes_combination]
            assert hasattr(classes_combination, '__iter__')
            assert len(classes_combination) <= 3
            ######################### root, super, sub
            if len(classes_combination) == 3:
                rootclass=classes_combination[0]
                superclass = classes_combination[-2]

                capped_superclass_data_pct = superclass_2_data_pct[superclass]
                capped_rootclass_data_pct = superclass_2_data_pct[rootclass]

                superclass_data_end = int(1.0 * capped_superclass_data_pct * len(data_subset_idx))
                superclass_data_start = int((1-subclass_data_pct) * capped_superclass_data_pct * len(data_subset_idx))

                rootclass_data_len = int(capped_rootclass_data_pct * capped_superclass_data_pct * len(data_subset_idx))

                subclass_data_len = int(subclass_data_pct * len(data_subset_idx))

                subclass = classes_combination[-1]

                if rootclass in cla_to_data_idx.keys():
                    cla_to_data_idx[rootclass].extend(data_subset_idx[:rootclass_data_len])
                else:
                    cla_to_data_idx[rootclass] = data_subset_idx[:rootclass_data_len]

                if superclass in cla_to_data_idx.keys():
                    cla_to_data_idx[superclass].extend(data_subset_idx[superclass_data_start:superclass_data_end])
                else:
                    cla_to_data_idx[superclass] = data_subset_idx[superclass_data_start:superclass_data_end]

                if subclass in cla_to_data_idx.keys():
                    cla_to_data_idx[subclass].extend(data_subset_idx[-subclass_data_len:])
                else:
                    cla_to_data_idx[subclass] = data_subset_idx[-subclass_data_len:]

            elif len(classes_combination) == 2:
                superclass = classes_combination[0]
                capped_superclass_data_pct = superclass_2_data_pct[superclass]
                superclass_data_len = int(capped_superclass_data_pct * len(data_subset_idx))
                subclass_data_len = int(subclass_data_pct * len(data_subset_idx))
                superclass = classes_combination[0]
                subclass = classes_combination[1]

                if superclass in cla_to_data_idx.keys():
                    cla_to_data_idx[superclass].extend(data_subset_idx[:superclass_data_len])
                else:
                    cla_to_data_idx[superclass] = data_subset_idx[:superclass_data_len]

                if subclass in cla_to_data_idx.keys():
                    cla_to_data_idx[subclass].extend(data_subset_idx[-subclass_data_len:])
                else:
                    cla_to_data_idx[subclass] = data_subset_idx[-subclass_data_len:]
            else:
                cla = classes_combination[0]
                if cla in cla_to_data_idx.keys():
                    cla_to_data_idx[cla].extend(data_subset_idx)
                else:
                    cla_to_data_idx[cla] = data_subset_idx

        task_id_to_data_idx = {task_id: [] for task_id in range(len(tasks))}
        for task_id in range(len(tasks)):
            task = tasks[task_id]
            for cla in task:
                if cla in cla_to_data_idx.keys():
                    task_id_to_data_idx[task_id].extend(cla_to_data_idx[cla])
            # Make sure there are no duplicates, and sort them to make sure samples are always having the same order,
            # as set() doesn't keep the order (keeping the order is useful to identify the sources of randomness)
            task_id_to_data_idx[task_id] = sorted(list(set(task_id_to_data_idx[task_id])))
        return task_id_to_data_idx

    def _divide_data_across_tasks(self) -> Dict[int, List[int]]:
        """
        Divide the dataset across the tasks depending on the classes per task.

        Returns:
            Dict[int, List[int]]: a dictionary with the task id as key, and the list of the samples that corespond to
            that task as the value
        """
        dataset = self.dataset
        tasks = self.tasks
        task_id_to_data_idx = {task_id: [] for task_id in range(len(tasks))}
        for task_id in range(len(tasks)):
            task = tasks[task_id]
            for cla in task:
                data_subset_idx = [sample_id for sample_id in range(len(dataset))
                                   if cla in dataset[sample_id][1]]
                task_id_to_data_idx[task_id].extend(data_subset_idx)
            # Make sure there are no duplicates, and sort them to make sure samples are always having the same order,
            # as set() doesn't keep the order (keeping the order is useful to identify the sources of randomness)
            task_id_to_data_idx[task_id] = sorted(list(set(task_id_to_data_idx[task_id])))
        return task_id_to_data_idx

    def _check_complete_information_mode(self):
        if self.setup == IIRC_SETUP:
            if self.test_mode and not self.complete_information_mode:
                warnings.warn("complete_information_mode is set to False for the current test set")
            elif not self.test_mode and self.complete_information_mode:
                warnings.warn("complete_information_mode is set to True for the current train/validation set")
        else:
            pass

    def choose_task(self, task_id: int) -> None:
        """
        Load the data corresponding to task "task_id" and update tbe seen classes based on it.

        Args:
            task_id (int): The task_id of the task to load
        """
        self._check_complete_information_mode()
        self.cur_task_id = task_id
        self.cur_task = self.tasks[self.cur_task_id]
        self._load_cur_task_data()
        self._update_seen_classes()
        self.cached_images = {}

    def load_tasks_up_to(self, task_id: int) -> None:
        """
        Load the data corresponding to the tasks up to "task_id" (including that task). When using the IIRC setup, this
            function is only available when complete_information_mode is set to True.

        Args:
            task_id (int): The task_id of the task to load
        """
        if self.setup == IIRC_SETUP:
            assert self.complete_information_mode, "load_tasks_up_to is only available during complete information mode"
        self.cur_task_id = task_id
        self.cur_task = [cla for task in self.tasks[:task_id + 1] for cla in task]
        self._load_data_up_to(task_id)
        self._update_seen_classes()
        self.cached_images = {}

    def _load_cur_task_data(self):
        self.cur_task_data_ids = self.task_id_to_data_idx[self.cur_task_id]

    def _load_data_up_to(self, task_id):
        data_ids = []
        for i in range(task_id + 1):
            data_ids.extend(self.task_id_to_data_idx[i])
        # remove duplicates (as tasks can have common samples with different labels)
        self.cur_task_data_ids = sorted(list(set(data_ids)))

    def _update_seen_classes(self):
        self.seen_classes = list(set(self.seen_classes) | set(self.cur_task))

    def __len__(self):
        return len(self.cur_task_data_ids)

    def get_labels(self, index: int) -> Tuple[str, str]:
        """
        Return the labels of the sample with index (index) in the current task.

        Args:
            index (int): The index of the sample in the current task, this is a relative index within the current task

        Returns:
            Tuple[str, str]: The labels corresponding to the sample. If using CIL setup, or if the other label is
            masked, then the other str contains the value specified by the NO_LABEL_PLACEHOLDER
        """
        sample_idx = self.cur_task_data_ids[index]
        _, labels = self.dataset[sample_idx]
        if self.setup == IIRC_SETUP:
            # mask classes outside the current task when not in complete information mode
            if not self.complete_information_mode:
                labels = list(set(labels) & set(self.cur_task))
            # mask unseen classes (classes from subsequent tasks)
            elif self.complete_information_mode:
                labels = list(set(labels) & set(self.seen_classes))
            if len(labels) ==1 :
                labels.append(NO_LABEL_PLACEHOLDER)
                labels.append(NO_LABEL_PLACEHOLDER)
            elif len(labels) == 2:
                labels.append(NO_LABEL_PLACEHOLDER)
            else:
                pass

        elif self.setup == CIL_SETUP:
            assert len(labels) == 1, "More than one label is set to True during Class Incremental Setup!"
            labels = list(labels)
            labels.append(NO_LABEL_PLACEHOLDER)

        assert len(labels) == 3
        return labels[0], labels[1], labels[2]

    def get_item(self, index: int) -> Tuple[Any, str, str]:
        """
        Return the image with index (index) in the current task along with its labels. No transformations are applied
            to the image.

        Args:
            index (int): The index of the sample in the current task, this is a relative index within the current task

        Returns:
            Tuple[Any, str, str]: The image along with its labels . If using CIL setup, or if the other label is masked,
            then the other str contains the value specified by the NO_LABEL_PLACEHOLDER
        """
        sample_idx = self.cur_task_data_ids[index]
        if self.using_image_path:
            image_path, labels = self.dataset[sample_idx]
            if image_path not in self.cached_images:
                try:
                    image = Image.open(image_path)
                    image = image.convert('RGB')
                except Exception as e:
                    iirc.utils.utils.print_msg(e)
                    return
                if self.cache_images:
                    self.cached_images[image_path] = image
            else:
                image = self.cached_images[image_path]
        else:
            image, labels = self.dataset[sample_idx]
        # assert that this sample shares some labels with the current task
        assert len(set(labels) & set(self.cur_task)) > 0

        if self.setup == IIRC_SETUP:
            # mask classes outside the current task when not in complete information mode
            if not self.complete_information_mode:
                labels = list(set(labels) & set(self.cur_task))
            # mask unseen classes (classes from subsequent tasks)
            elif self.complete_information_mode:
                labels = list(set(labels) & set(self.seen_classes))
            if len(labels) ==1 :
                labels.append(NO_LABEL_PLACEHOLDER)
                labels.append(NO_LABEL_PLACEHOLDER)
            elif len(labels)==2:
                labels.append(NO_LABEL_PLACEHOLDER)
            else:
                pass

        elif self.setup == CIL_SETUP:
            assert len(labels) == 1, "More than one label is set to True during Class Incremental Setup!"
            labels = list(labels)
            labels.append(NO_LABEL_PLACEHOLDER)

        assert len(labels) == 3
        return image, labels[0], labels[1], labels[2]

    @abstractmethod
    def __getitem__(self, index):
        pass

    def get_image_indices_by_cla(self, cla: str, num_samples: int = -1, shuffle: bool = True) -> np.ndarray:

        assert cla in self.cur_task
        cla_samples_idx = np.array([i for i in range(len(self.cur_task_data_ids))
                                    if cla in self.dataset[self.cur_task_data_ids[i]][1]])
        # cla_samples_idx = np.nonzero(self.labels == label)[0]
        if shuffle:
            np.random.shuffle(cla_samples_idx)
        if len(cla_samples_idx) > num_samples != -1:
            return cla_samples_idx[:num_samples]
        else:
            return cla_samples_idx

    @contextmanager
    def disable_augmentations(self) -> None:
        """A context where only the essential transformations are applied"""
        cur_augmentations_state = self._apply_augmentations
        self._apply_augmentations = False
        try:
            yield
        finally:
            self._apply_augmentations = cur_augmentations_state

    def enable_complete_information_mode(self) -> None:
        self.complete_information_mode = True

    def enable_incomplete_information_mode(self) -> None:
        self.complete_information_mode = False
