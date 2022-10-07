import atexit
import copy
import os
import shutil

import numpy as np
import omegaconf
from omegaconf import OmegaConf


class OptunaConfigConverter:
    def __init__(self, path="config/base.yaml", logdir=None):
        self.cfg = OmegaConf.load(path)
        self.study = None
        self.logdir = logdir
        self.cur_cfg = None
        self.all_cfgs = []
        self.all_scores = []

        new_cfg_path = os.path.join(logdir, "train.yaml")
        if not os.path.exists(new_cfg_path):
            shutil.copy(path, new_cfg_path)

        atexit.register(self.make_test_cfg)

    def hook_study(self, study):
        self.study = study

    def register_suggestion(self, cfg, trial):
        if type(cfg) == omegaconf.dictconfig.DictConfig:
            if "dist" in cfg:
                dist = cfg.pop("dist")
                suggested_value = getattr(trial, dist)(**cfg)
                return suggested_value
            for key, param_cfg in cfg.items():
                cfg[key] = self.register_suggestion(param_cfg, trial)
        return cfg

    def suggest_params(self, trial):
        cfg = copy.deepcopy(self.cfg)
        cfg = self.register_suggestion(cfg, trial)

        self.cur_cfg = cfg
        self.all_cfgs.append(cfg)
        return cfg

    def make_test_cfg(self):
        trials = self.study.trials
        best_trial_idx = np.argmax([trial.value for trial in trials])
        best_cfg = self.all_cfgs[best_trial_idx]
        test_cfg_path = os.path.join(self.logdir, "test.yaml")
        OmegaConf.save(best_cfg, test_cfg_path)

    def format_cfg(self):
        text = ""
        text += "\n#=========== Starting with params of ... ===========#\n"
        for param_key, param_value in self.cur_cfg.items():
            text += "{:<14} = {}\n".format(param_key, param_value)
        return text
