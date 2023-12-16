import math
import re
import subprocess
import time
import unittest

checkpoint_dir = "test_checkpoints"
base_args = "--log_reduced_training_loss 1 "
data_args = "--training_dir /fsx/datasets/train_ids_wsvocab_redo_2048_smaller --test_dir /fsx/datasets/val_ids_wsvocab_2048 "

MODELS = [
    ("gpt_neox", None),
    ("llama_v2", None),
    ("llama_v2", "/fsx/users/huilgolr/hf_pretrained_models/Llama-2-7b-hf"),
]
USE_ORIG_PARAMS = [0, 1]
freq_args = "--validation_freq 10 --checkpoint_freq 10 --logging_freq 1 "
from dataclasses import dataclass

# this is used to debug for a given list of job log files
# SIMULATED = [22223,22224,22225,22226,22227,22228, 22229,22230, 22231, 22232, 22233, 22234]
SIMULATED = []


@dataclass
class Job:
    """Class for keeping track of an item in inventory."""

    job_name: str
    slurm_job_id: int
    status: str = "PENDING"
    done: bool = False

    def get_log_file(self):
        return f"logs/{self.job_name}_{self.slurm_job_id}.out"


class TestModelSaveLoad(unittest.TestCase):
    def build_job_cmd(
        self, model_name, size, pretrained, use_orig_param, save_job, depends_job_id=None
    ):
        finetune = 0 if pretrained is None else 1
        save_or_load = "save" if save_job else "load"
        job_name = f"{model_name}_pretrained{finetune}_origparam{use_orig_param}_{save_or_load}"
        cmd = f"sbatch -N 1 --parsable --job-name {job_name} --partition benchmark "
        if depends_job_id:
            cmd += f"--dependency=afterok:{depends_job_id} "
        cmd += "smprun -v2 -conda -env /fsx/users/huilgolr/conda/rubik2dev scripts/model.sh "

        cmd += base_args + data_args + freq_args
        cmd += f"--model_type {model_name} --model_size {size} --checkpoint_dir {checkpoint_dir}/{job_name} "
        if pretrained:
            cmd += f"--hf_pretrained_model_name_or_dir {pretrained} "
        if use_orig_param:
            cmd += f"--use_orig_params {use_orig_param} "
        if save_job:
            cmd += "--max_steps 50 "
        else:
            save_job_name = job_name.replace("load", "save")
            cmd += f"--max_steps 60 --resume_from_checkpoint {checkpoint_dir}/{save_job_name}/{model_name}-40steps --checkpoint_freq 100 "

        return cmd, job_name

    def kickoff_job(self, cmd, job_name):

        if SIMULATED:
            slurm_job_id = SIMULATED[self.ctr]
        else:
            res = subprocess.run(cmd, shell=True, capture_output=True)
            slurm_job_id = res.stdout.decode("utf-8").strip()

        job = Job(job_name=job_name, slurm_job_id=slurm_job_id)
        self.ctr += 1
        print(f"Started {job} with logfile {job.get_log_file()}")
        return job

    def test(self):
        jobs = []
        pairs = []
        self.ctr = 0
        for model, pretrained in MODELS:
            for use_orig_param in USE_ORIG_PARAMS:
                cmd, job_name = self.build_job_cmd(
                    model, "7b", pretrained, use_orig_param, save_job=True
                )
                save_job = self.kickoff_job(cmd, job_name)
                jobs.append(save_job)
                cmd, job_name = self.build_job_cmd(
                    model,
                    "7b",
                    pretrained,
                    use_orig_param,
                    save_job=False,
                    depends_job_id=save_job.slurm_job_id,
                )
                resume_job = self.kickoff_job(cmd, job_name)

        jobs.append(resume_job)

        pairs.append((save_job, resume_job))

        # wait for all job ids
        self.wait_all(jobs)

        # verify loss
        for pair in pairs:
            self.verify_loss(pair[0].get_log_file(), pair[1].get_log_file())

    def get_losses(self, filepath):
        losses = {}
        with open(filepath, "r") as f:
            line = f.readline()
            while line:
                expr = "Batch (\\d+) Loss: ([+-]?[0-9]*[.]?[0-9]+)"
                matches = re.search(expr, line)
                if matches:
                    losses[matches.group(1)] = float(matches.group(2))
                line = f.readline()
        return losses

    def verify_loss(self, save_job_file, load_job_file):
        print("Verifying losses for ", save_job_file, load_job_file)
        save_losses = self.get_losses(save_job_file)
        load_losses = self.get_losses(load_job_file)
        for k in load_losses.keys():
            if k in save_losses:
                assert math.isclose(save_losses[k], load_losses[k], abs_tol=1e-1), (
                    k,
                    save_losses[k],
                    load_losses[k],
                )
                print("Loss matched at step", k)

    def wait_all(self, jobs):
        time.sleep(2)
        all_jobs = jobs.copy()
        num_done = 0
        while num_done < len(jobs):
            num_done = 0
            time.sleep(1)
            for j in all_jobs:
                if not j.done:
                    res = subprocess.run(
                        f"sacct -j {j.slurm_job_id} -n -o state | head -n 1",
                        shell=True,
                        capture_output=True,
                    )
                    j.status = res.stdout.decode("utf-8").strip()
                    if j.status and j.status not in ["PENDING", "RUNNING", "REQUEUED", "SUSPENDED"]:
                        # done
                        j.done = True
                        num_done += 1
                        print(j)
                else:
                    num_done += 1


if __name__ == "__main__":
    unittest.main()
