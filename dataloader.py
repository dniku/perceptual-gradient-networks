import io
import logging
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.utils.data
from PIL import Image
from torchvision import datasets, transforms


class RandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly. If without replacement, shuffles the dataset and takes the first
    :attr:`num_samples`. If with replacement, simply generates :attr:`num_samples` random indices.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (torch.Generator): optional, generator to use for generating indices.
        log_indices (bool): log generated indices with Python's `logging` module.
    """

    def __init__(self, data_source, replacement=False, num_samples=None, generator=None, log_indices=False):
        super().__init__(data_source)

        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self.log_indices = log_indices

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)

        if self.replacement:
            if self.generator is not None:
                indices = torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64, generator=self.generator)
            else:
                indices = torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64)
        else:
            if self.generator is not None:
                indices = torch.randperm(n, generator=self.generator)
            else:
                indices = torch.randperm(n)
            if self._num_samples is not None:
                indices = indices[:self._num_samples]
        indices = indices.tolist()

        if self.log_indices:
            logging.debug(f'Sampled {len(indices)} indices: {indices}')

        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedRandomSampler(RandomSampler):
    def __init__(self, data_source, replacement=False, num_samples=None, log_indices=False, rank0_seed=0, rank=None):
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        generator = torch.Generator()
        generator.manual_seed(rank0_seed + rank)

        super().__init__(data_source, replacement, num_samples, generator, log_indices)


class RetryingFileReader:
    def __init__(self, attempts=60, retry_delay=1):
        self.attempts = attempts
        self.retry_delay = retry_delay

    def __call__(self, path):
        for j in range(self.attempts):
            try:
                with open(path, 'rb') as fp:
                    yield fp.read()
            except OSError as e:
                logging.error(f'Attempt {j}/{self.attempts}: failed to read binary data {path}:\n{e}')
                if j == self.attempts - 1:
                    raise
                else:
                    time.sleep(self.retry_delay)


class PilLoader:
    def __init__(self, get_bytes_by_path):
        self.get_bytes_by_path = get_bytes_by_path

    def __call__(self, path):
        for j, data in enumerate(self.get_bytes_by_path(path), 1):
            with io.BytesIO(data) as fp:
                img = Image.open(fp)
                try:
                    return img.convert('RGB')
                except OSError as e:
                    logging.error(f'Attempt {j}: failed to decode image file {path}:\n{e}')
        raise RuntimeError('Could not decode image file')


class StyleganPretuned(torch.utils.data.Dataset):
    def __init__(self, root: Path):
        self.root = root

    @classmethod
    def _get_batch_element(cls, a, idx):
        if isinstance(a, torch.Tensor):
            return a[idx]
        elif isinstance(a, int) or isinstance(a, float) or isinstance(a, bool):
            return a
        elif isinstance(a, dict):
            return {k: cls._get_batch_element(v, idx) for k, v in a.items()}
        elif isinstance(a, list):
            return [cls._get_batch_element(v, idx) for v in a]
        elif isinstance(a, tuple):
            return tuple(cls._get_batch_element(v, idx) for v in a)
        else:
            assert False

    def __getitem__(self, index):
        assert 0 <= index < 100
        batch = torch.load(self.root / f'{index // 25:03d}.pth', map_location='cpu')
        assert isinstance(batch, dict) and set(batch.keys()) >= {'latents', 'images'}
        batch = {
            'latents': self._get_batch_element(batch['latents'], index % 25),
            'images': batch['images'][index % 25],
            # Collating opt_state properly is hard
            # 'opt_state': self._get_batch_element(batch['opt_state'], index % 25),
        }
        return batch, 0

    def __len__(self):
        return 1000


def get_dataloaders(dataset_name: str, data_dir: Path,
                    train_batch_size: int, valid_batch_size: int,
                    *,
                    batches_per_train_epoch=None, batches_per_valid_epoch=None, replacement=True,
                    valid_first_samples=None,
                    train_num_workers=0,
                    resize_to=256, crop_to=224, normalize=True,
                    distributed=False):
    to_torch = [transforms.ToTensor()]
    if normalize:
        to_torch.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ))

    get_bytes_by_path = RetryingFileReader()
    loader = PilLoader(get_bytes_by_path)

    if dataset_name.lower() == 'imagenet':
        imagenet_root = data_dir / 'ImageNet'

        logging.debug(f'Initializing train dataset in {imagenet_root}...')
        train_dataset = datasets.ImageNet(
            str(imagenet_root),
            split='train',
            transform=transforms.Compose([
                transforms.RandomResizedCrop(crop_to),
                transforms.RandomHorizontalFlip(),
                *to_torch,
            ]),
            loader=loader,
        )

        logging.debug(f'Initializing valid dataset in {imagenet_root}...')
        valid_dataset = datasets.ImageNet(
            str(imagenet_root),
            split='val',
            transform=transforms.Compose([
                transforms.Resize(resize_to),
                transforms.CenterCrop(crop_to),
                *to_torch,
            ]),
            loader=loader,
        )
    elif dataset_name.lower() == 'ffhq':
        ffhq_root = data_dir / 'ffhq-dataset/images1024x1024'

        def is_train(path: Path):
            return int(Path(path).name[:-len('.png')]) < 60000

        def is_valid(path: Path):
            return int(Path(path).name[:-len('.png')]) >= 60000

        logging.debug('Initializing train dataset...')
        train_dataset = datasets.ImageFolder(
            str(ffhq_root),
            transform=transforms.Compose([
                transforms.RandomResizedCrop(crop_to),
                transforms.RandomHorizontalFlip(),
                *to_torch,
            ]),
            is_valid_file=is_train,
            loader=loader,
        )

        logging.debug('Initializing valid dataset...')
        valid_dataset = datasets.ImageFolder(
            str(ffhq_root),
            transform=transforms.Compose([
                transforms.Resize(resize_to),
                transforms.CenterCrop(crop_to),
                *to_torch,
            ]),
            is_valid_file=is_valid,
            loader=loader,
        )
    elif dataset_name.lower() == 'stylegan_pretuned':
        stylegan_pretuned_root = data_dir / 'stylegan_pretuned'
        train_dataset = valid_dataset = StyleganPretuned(stylegan_pretuned_root)
    else:
        raise NotImplementedError(f'Unsupported dataset: {dataset_name}')

    sampler_cls = DistributedRandomSampler if distributed else RandomSampler

    train_num_samples = batches_per_train_epoch * train_batch_size if batches_per_train_epoch is not None else None
    train_sampler = sampler_cls(
        train_dataset, replacement=replacement, num_samples=train_num_samples)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size,
        num_workers=train_num_workers,
        sampler=train_sampler,
        pin_memory=True,
    )

    valid_num_samples = batches_per_valid_epoch * valid_batch_size if batches_per_valid_epoch is not None else None
    if valid_first_samples is not None:
        if distributed:
            valid_sampler = torch.utils.data.distributed.DistributedSampler(range(valid_first_samples), shuffle=False)
        else:
            valid_sampler = torch.utils.data.SequentialSampler(range(valid_first_samples))
    else:
        valid_sampler = sampler_cls(
            valid_dataset, replacement=replacement, num_samples=valid_num_samples, log_indices=True)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batch_size,
        num_workers=1 if train_num_workers > 0 else 0,
        sampler=valid_sampler,
        pin_memory=True,
    )

    return train_dataloader, valid_dataloader
