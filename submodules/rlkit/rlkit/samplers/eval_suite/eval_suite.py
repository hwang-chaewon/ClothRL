from typing import List
from torch.serialization import save
from rlkit.samplers.eval_suite.utils import get_obs_preprocessor


class EvalTest(object):
    def __init__(self, env, policy, keys, name, metric_keys, num_runs, save_images_every_epoch, max_path_length, additional_keys, save_blurred_images, frame_stack_size, save_folder):
        self.policy = policy
        self.name = name
        self.epoch = 0
        self.metric_keys = metric_keys

        self.save_images_every_epoch = save_images_every_epoch
        self.max_path_length = max_path_length
        self.obs_preprocessor = get_obs_preprocessor(
            keys['path_collector_observation_key'], additional_keys, keys['desired_goal_key'])
        self.save_blurred_images = save_blurred_images

        self.env = env
        self.num_runs = num_runs
        self.frame_stack_size = frame_stack_size
        self.base_save_folder = save_folder

    def run_evaluations(self) -> float:
        results = dict()
        for metric_key in self.metric_keys:
            results[metric_key] = 0

        for i in range(self.num_runs):
            print("============== [3] eval step", i, " / ", self.num_runs, "=====================")
            result = self.single_evaluation(i)
            for result_key in result.keys():
                results[result_key] += result[result_key]/self.num_runs

        return results

    def single_evaluation(self) -> float:
        raise NotImplementedError("You need to implement single_evaluation")

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class EvalTestSuite(object):
    def __init__(self, tests: List[EvalTest]):
        self.tests = tests

    def run_evaluations(self, epoch: int) -> dict:
        results = dict()
        for test in self.tests:
            test.set_epoch(epoch)
            metrics = test.run_evaluations()
            for metric_key in metrics.keys():
                results[f"{test.name}_{metric_key}"] = metrics[metric_key]
        return results
    