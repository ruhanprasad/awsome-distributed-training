import unittest

import torch
import torch.distributed as dist

dist.init_process_group("gloo")
from data.pipelines import GPTDataPipeline, HFDataPipeline


def run_through_loader(loader, batch_fn, num_batches, size_for_input):
    """
    This function runs through dataloader passed, and caches the samples produced by data pipeline.
    It also asserts that all batches match the size_for_inptu passed
    """
    samples = []
    for i, data in enumerate(loader):
        try:
            input_ids, mask, labels = batch_fn(data)
        except ValueError:
            input_ids, mask = batch_fn(data)
            labels = None
        assert input_ids.shape == size_for_input, (input_ids.shape, size_for_input)
        assert mask.shape == size_for_input
        if labels is not None:
            assert labels.shape == size_for_input
        samples.append((input_ids, mask, labels))
        if i == num_batches:
            break
    return samples


def compare_with_samples(resume_loader, batch_fn, train_samples, num_batches, start_batch):
    """
    This function runs through dataloader passed and compares them to the samples passed
    """
    for i, data in enumerate(resume_loader):
        input_ids, mask, labels = batch_fn(data)
        if i == num_batches:
            break
        assert torch.equal(input_ids, train_samples[start_batch + i][0])
        assert torch.equal(mask, train_samples[start_batch + i][1])
        assert torch.equal(labels, train_samples[start_batch + i][2])


class TestHFSequence(unittest.TestCase):
    def test_wikicorpus(
        self, seqlen=4096, train_batch_size=4, val_batch_size=2, num_batches=10, resume_batches=5
    ):
        self.basic_test_dataloader(
            "/fsx/datasets/wikicorpus__raw_en/llama/4096/train/",
            "/fsx/datasets/wikicorpus__raw_en/llama/4096/val/",
            seqlen=seqlen,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            num_batches=num_batches,
            resume_batches=resume_batches,
        )

    def test_c4(
        self, seqlen=4096, train_batch_size=4, val_batch_size=2, num_batches=10, resume_batches=5
    ):
        self.basic_test_dataloader(
            "/fsx/datasets/c4/en/hf-tokenized/llama/train/",
            "/fsx/datasets/c4/en/hf-tokenized/llama/val/",
            seqlen=seqlen,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            num_batches=num_batches,
            resume_batches=resume_batches,
        )

    def basic_test_dataloader(
        self,
        train_path,
        val_path,
        seqlen=4096,
        train_batch_size=4,
        val_batch_size=2,
        num_batches=10,
        resume_batches=5,
    ):
        pipeline = HFDataPipeline(
            train_path,
            train_batch_size=train_batch_size,
            dataset_val_path=val_path,
            val_batch_size=val_batch_size,
            dp_rank=dist.get_rank(),
            dp_size=dist.get_world_size(),
            shuffle=False,
        )
        train_samples = run_through_loader(
            pipeline.train_dataloader,
            pipeline.get_batch,
            num_batches=num_batches,
            size_for_input=torch.Size([train_batch_size, seqlen]),
        )
        run_through_loader(
            pipeline.val_dataloader,
            pipeline.get_val_batch,
            num_batches=num_batches,
            size_for_input=torch.Size([val_batch_size, seqlen]),
        )
        pipeline.resume_from_sequence_number = 5 * train_batch_size
        resume_loader = pipeline._create_dataloader(
            pipeline.train_dataset, pipeline.train_batch_size
        )
        compare_with_samples(
            resume_loader,
            pipeline.get_batch,
            train_samples=train_samples,
            num_batches=resume_batches,
            start_batch=num_batches - resume_batches,
        )


class TestGPTSequence(unittest.TestCase):
    def test_orig_dataset(self, seqlen=2048, train_batch_size=4, num_batches=10, resume_batches=5):
        self.basic_test_dataloader(
            "/fsx/datasets/train_ids_wsvocab_redo_2048_smaller",
            seqlen=seqlen,
            batch_size=train_batch_size,
            num_batches=num_batches,
            resume_batches=resume_batches,
        )

    def test_orig_dataset_with_path_index(
        self, seqlen=2048, train_batch_size=4, num_batches=10, resume_batches=5
    ):
        self.basic_test_dataloader(
            "/fsx/datasets/train_ids_wsvocab_redo_2048_smaller",
            seqlen=seqlen,
            batch_size=train_batch_size,
            path_index=5,
            num_batches=num_batches,
            resume_batches=resume_batches,
        )

    def basic_test_dataloader(
        self,
        path,
        seqlen=4096,
        batch_size=4,
        num_batches=10,
        resume_batches=5,
        path_index=0,
    ):
        pipeline = GPTDataPipeline(
            path,
            train_batch_size=batch_size,
            dp_rank=dist.get_rank(),
            dp_size=dist.get_world_size(),
            start_path_index=path_index,
            zipped_data=True,
            shuffle=False,
        )
        pipeline.create_train_dataset()

        train_samples = run_through_loader(
            pipeline.train_dataloader,
            pipeline.get_batch,
            num_batches=num_batches,
            size_for_input=torch.Size([batch_size, seqlen]),
        )
        pipeline.resume_from_sequence_number = 5 * batch_size
        pipeline.create_train_dataset()
        compare_with_samples(
            pipeline.train_dataloader,
            pipeline.get_batch,
            train_samples=train_samples,
            num_batches=resume_batches,
            start_batch=num_batches - resume_batches,
        )


if __name__ == "__main__":
    unittest.main()
