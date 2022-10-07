import atexit
import csv
import logging
import os
import subprocess
from datetime import datetime

import git

_HEADERS = ["command", "logdir", "git SHA", "message", "val score", "test score"]


def setup_logger(logdir, file_name="train.log", master_log=None):
    log_level = logging.INFO
    if os.path.exists(logdir):
        print("Overwrite existing logging dir !")
    else:
        os.makedirs(logdir)
    log_path = os.path.join(logdir, file_name)

    fmt = "%(asctime)s :%(message)s"
    logging.basicConfig(level=log_level, filename=log_path, format=fmt)
    logger = logging.getLogger()
    return logger


class ExperimetManager:
    def __init__(
        self, path, logdir, study=None, sys_args=None, message=None, write_md=False
    ):
        if not os.path.exists(path):
            with open(path, "w") as f:
                separator_dict = {}
                for head in _HEADERS:
                    separator_dict[head] = "--"
                writer = csv.DictWriter(f, fieldnames=_HEADERS, delimiter="|")
                writer.writeheader()

        self.study = study
        self.path = path
        self.sys_args = sys_args
        self.logdir = logdir
        self.message = message

        if write_md:
            atexit.register(self.logging_md)

    def logging_md(self):
        expt = {}
        expt["command"] = self.retrieve_command(self.sys_args)
        expt["logdir"] = self.logdir
        expt["val score"] = "{:2f}".format(self.study.best_value)
        expt["test score"] = ""
        expt["git SHA"] = get_git_sha()
        expt["message"] = self.message

        with open(self.path, "r") as f:
            csv_file = csv.DictReader(f, delimiter="|")
            fieldnames = csv_file.fieldnames

        with open(self.path, "a") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="|")
            writer.writerow(expt)

    @staticmethod
    def retrieve_command(sys_args):
        command = "python "
        for arg in sys_args:
            command += "{} ".format(arg)
        return command

    def insert_item(self, key, value, identifier="logdir"):
        with open(self.path, "r") as f:
            csv_file = csv.DictReader(f, delimiter="|")
            fieldnames = csv_file.fieldnames
            new_lines = []
            for line in csv_file:
                if line[identifier] == self.__getattribute__(identifier):
                    line[key] = value
                new_lines.append(line)
        with open(self.path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="|")
            writer.writeheader()
            writer.writerows(new_lines)


def get_git_sha():
    repo = git.Repo(search_parent_directories=True)
    git_sha = repo.head.object.hexsha
    return git_sha[:8]


def log_vcs(logger, logdir):
    try:
        repo = git.Repo(search_parent_directories=True)
        git_date = datetime.fromtimestamp(repo.head.object.committed_date).strftime(
            "%Y-%m-%d"
        )
        git_sha = repo.head.object.hexsha
        git_message = repo.head.object.message
        logger.info(
            "Source is from Commit {} ({}): {}".format(
                git_sha[:8], git_date, git_message.strip()
            )
        )

        # Also create diff file in the log directory
        if logdir is not None:
            with open(os.path.join(logdir, "compareHead.diff"), "w") as fid:
                subprocess.run(["git", "diff"], stdout=fid)

    except git.exc.InvalidGitRepositoryError:
        pass
