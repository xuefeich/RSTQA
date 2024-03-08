import functools
import itertools
import logging
import os
import queue
import threading
import warnings

from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union

import multiprocessing as python_multiprocessing
import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
import torch.utils.data.graph_settings

from torch._utils import ExceptionWrapper

from torch.utils.data import (
    IterDataPipe,
    MapDataPipe,
    IterableDataset,
    Sampler,
    SequentialSampler,
    RandomSampler,
    BatchSampler,
    Dataset,)

from torch.utils.data.datapipes.datapipe import _IterDataPipeSerializationWrapper, _MapDataPipeSerializationWrapper

from torch.utils.data import _utils

__all__ = [
    "DataLoader",
    "get_worker_info",
    "default_collate",
    "default_convert",
]

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
_worker_init_fn_t = Callable[[int], None]

# Ideally we would parameterize `DataLoader` by the return type of `collate_fn`, but there is currently no way to have that
# type parameter set to a default value if the user doesn't pass in a custom 'collate_fn'.
# See https://github.com/python/mypy/issues/3737.
_collate_fn_t = Callable[[List[T]], Any]


# These functions used to be defined in this file. However, it was moved to
# _utils/collate.py. Although it is rather hard to access this from user land
# (one has to explicitly directly `import torch.utils.data.dataloader`), there
# probably is user code out there using it. This aliasing maintains BC in this
# aspect.
default_collate: _collate_fn_t = _utils.collate.default_collate
default_convert = _utils.collate.default_convert

get_worker_info = _utils.worker.get_worker_info

logger = logging.getLogger(__name__)


class _DatasetKind:
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        else:
            return _utils.fetch._IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)


class _InfiniteConstantSampler(Sampler):
    r"""Analogous to ``itertools.repeat(None, None)``.

    Used as sampler for :class:`~torch.utils.data.IterableDataset`.
    """

    def __iter__(self):
        while True:
            yield None


def _get_distributed_settings():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    else:
        return 1, 0


def _sharding_worker_init_fn(worker_init_fn, world_size, rank_id, worker_id):
    global_worker_id = worker_id
    info = torch.utils.data.get_worker_info()
    assert info is not None
    total_workers = info.num_workers
    datapipe = info.dataset
    assert isinstance(datapipe, (IterDataPipe, MapDataPipe))
    # To distribute elements across distributed process evenly, we should shard data on distributed
    # processes first then shard on worker processes
    total_workers *= world_size
    global_worker_id = global_worker_id * world_size + rank_id
    # For BC, use default SHARDING_PRIORITIES
    torch.utils.data.graph_settings.apply_sharding(datapipe, total_workers, global_worker_id)
    if worker_init_fn is not None:
        worker_init_fn(worker_id)


def _share_dist_seed(generator, pg):
    _shared_seed = torch.empty((), dtype=torch.int64).random_(generator=generator)
    if isinstance(pg, dist.ProcessGroup):
        dist.broadcast(_shared_seed, src=0, group=pg)
    return _shared_seed.item()


class DataLoader(Generic[T_co]):
    batch_size: Optional[int]
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    sampler: Union[Sampler, Iterable]
    pin_memory_device: str
    prefetch_factor: Optional[int]
    _iterator : Optional['_BaseDataLoaderIter']
    __initialized = False

    def __init__(self, batch_size: Optional[int] = 1, num_ops = 6,
                 shuffle: Optional[bool] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None,
                 *, prefetch_factor: Optional[int] = None,
                 persistent_workers: bool = False,
                 pin_memory_device: str = ""):
        torch._C._log_api_usage_once("python.data_loader")

        if num_workers > 0 and prefetch_factor is None:
            prefetch_factor = 2
        elif prefetch_factor is not None and prefetch_factor < 0:
            raise ValueError('prefetch_factor option should be non-negative')

        if persistent_workers and num_workers == 0:
            raise ValueError('persistent_workers option needs num_workers > 0')

        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.batch_size = batch_size

        if collate_fn is None:
            if self._auto_collation:
                collate_fn = _utils.collate.default_collate
            else:
                collate_fn = _utils.collate.default_convert

        self.collate_fn = collate_fn
        self.persistent_workers = persistent_workers

        self.__initialized = True
        self._IterableDataset_len_called = None  # See NOTE [ IterableDataset and __len__ ]

        self._iterator = None

        self.check_worker_number_rationality()

        torch.set_vital('Dataloader', 'enabled', 'True')  # type: ignore[attr-defined]


        dpath =  f"tagop_llama_cached_train.pkl"
        self.num_ops = num_ops
        with open(dpath, 'rb') as f:
            print("Load data from {}.".format(dpath))
            data = pickle.load(f)

        all_data = []
        for item in data:
            input_ids = torch.from_numpy(item["input_ids"])
            attention_mask = torch.from_numpy(item["attention_mask"])
            token_type_ids = torch.from_numpy(item["token_type_ids"])
            paragraph_mask = torch.from_numpy(item["paragraph_mask"])
            table_mask = torch.from_numpy(item["table_mask"])
            paragraph_numbers = item["paragraph_number_value"]
            table_cell_numbers = item["table_cell_number_value"]
            paragraph_index = torch.from_numpy(item["paragraph_index"])
            table_cell_index = torch.from_numpy(item["table_cell_index"])
            tag_labels = torch.from_numpy(item["tag_labels"])
            operator_labels = torch.tensor(item["operator_label"])
            scale_labels = torch.tensor(item["scale_label"])
            #round_labels = torch.tensor(item["round_label"])
            gold_answers = item["answer_dict"]
            paragraph_tokens = item["paragraph_tokens"]
            table_cell_tokens = item["table_cell_tokens"]
            question_id = item["question_id"]
            opt_mask = item["opt_mask"]
            ari_ops = item["ari_ops"]
            opt_labels = item["opt_labels"]
            ari_labels = item["ari_labels"]
            selected_indexes = item["selected_indexes"]

            order_labels = item["order_labels"]
            question_mask = torch.from_numpy(item["question_mask"])

            opd_ids = torch.from_numpy(item["opd_ids"])
            opd_mask = torch.from_numpy(item["opd_mask"])
            
            all_data.append((input_ids, attention_mask, token_type_ids, paragraph_mask, table_mask, paragraph_index,
                table_cell_index, tag_labels, operator_labels, scale_labels, gold_answers,
                paragraph_tokens, table_cell_tokens, paragraph_numbers, table_cell_numbers,
                question_id,ari_ops,opt_labels,ari_labels,opt_mask,order_labels,selected_indexes,question_mask,
                opd_ids,opd_mask
                #,round_labels
                ))
        print("Load data size {}.".format(len(all_data)))
        self.data = TaTQABatchGen.make_batches(all_data, args.batch_size if self.is_train else args.eval_batch_size,
                                              self.is_train)
        self.offset = 0
        self.all_data = all_data
    
    @property
    def multiprocessing_context(self):
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        if multiprocessing_context is not None:
            if self.num_workers > 0:
                if isinstance(multiprocessing_context, str):
                    valid_start_methods = multiprocessing.get_all_start_methods()
                    if multiprocessing_context not in valid_start_methods:
                        raise ValueError(
                            'multiprocessing_context option '
                            f'should specify a valid start method in {valid_start_methods!r}, but got '
                            f'multiprocessing_context={multiprocessing_context!r}')
                    multiprocessing_context = multiprocessing.get_context(multiprocessing_context)

                if not isinstance(multiprocessing_context, python_multiprocessing.context.BaseContext):
                    raise TypeError('multiprocessing_context option should be a valid context '
                                    'object or a string specifying the start method, but got '
                                    f'multiprocessing_context={multiprocessing_context}')
            else:
                raise ValueError('multiprocessing_context can only be used with '
                                 'multi-process loading (num_workers > 0), but got '
                                 f'num_workers={self.num_workers}')

        self.__multiprocessing_context = multiprocessing_context

    def __setattr__(self, attr, val):
        if self.__initialized and attr in (
                'batch_size', 'batch_sampler', 'sampler', 'drop_last', 'dataset', 'persistent_workers'):
            raise ValueError(f'{attr} attribute should not be set after {self.__class__.__name__} is initialized')

        super().__setattr__(attr, val)

    def __iter__(self) -> '_BaseDataLoaderIter':
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            return self._get_iterator()

    @property
    def _auto_collation(self):
        return self.batch_sampler is not None

    def __len__(self) -> int:
        return len(self.data)

    def check_worker_number_rationality(self):
        def _create_warning_msg(num_worker_suggest, num_worker_created, cpuset_checked):

            suggested_max_worker_msg = ((
                "Our suggested max number of worker in current system is {}{}, which is smaller "
                "than what this DataLoader is going to create.").format(
                    num_worker_suggest,
                    ("" if cpuset_checked else " (`cpuset` is not taken into account)"))
            ) if num_worker_suggest is not None else (
                "DataLoader is not able to compute a suggested max number of worker in current system.")

            warn_msg = (
                "This DataLoader will create {} worker processes in total. {} "
                "Please be aware that excessive worker creation might get DataLoader running slow or even freeze, "
                "lower the worker number to avoid potential slowness/freeze if necessary.").format(
                    num_worker_created,
                    suggested_max_worker_msg)
            return warn_msg

        if not self.num_workers or self.num_workers == 0:
            return

        # try to compute a suggested max number of worker based on system's resource
        max_num_worker_suggest = None
        cpuset_checked = False
        if hasattr(os, 'sched_getaffinity'):
            try:
                max_num_worker_suggest = len(os.sched_getaffinity(0))
                cpuset_checked = True
            except Exception:
                pass
        if max_num_worker_suggest is None:
            # os.cpu_count() could return Optional[int]
            # get cpu count first and check None in order to satisfy mypy check
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                max_num_worker_suggest = cpu_count

        if max_num_worker_suggest is None:
            warnings.warn(_create_warning_msg(
                max_num_worker_suggest,
                self.num_workers,
                cpuset_checked))
            return

        if self.num_workers > max_num_worker_suggest:
            warnings.warn(_create_warning_msg(
                max_num_worker_suggest,
                self.num_workers,
                cpuset_checked))



