#!/usr/bin/python3

import argparse
import logging
import os
import psutil
import subprocess
import sys
import tempfile
import time
import urllib.request  # downloading

from io import BytesIO  # extracting zip without storing a file
from zipfile import ZipFile  # extracting a zip

# Create logger
log = logging.getLogger(__name__)

VERSION = "0.0.1"  # Version of this tool


def measure_call(call):
    """Run the given command, return measured performance data."""
    start_wallclock = time.perf_counter_ns()
    start_cpuclock = time.process_time_ns()
    # run command, ignore output
    # TODO extract status code of call
    _ = subprocess.run(call, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    end_wallclock = time.perf_counter_ns()
    end_cpuclock = time.process_time_ns()

    return {  # TODO: return status code of call
        "wall_time_ns": end_wallclock - start_wallclock,
        "cpu_time_ns": end_cpuclock - start_cpuclock
    }


class CNFgenerator(object):

    URL = "http://fmv.jku.at/cnfuzzdd/cnfuzzdd2013.zip"
    NAME = "cnfuzz"

    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info('creating an instance of Auxiliary')
        self.solver = None
        self.workdir = tempfile.TemporaryDirectory(
            prefix="specsat_solver", dir=os.getcwd())
        self.sourcefile = os.path.join(self.workdir.name, "cnfuzz.c")
        self.generator = os.path.join(self.workdir.name, self.NAME)
        self.log.debug("Get generator with workdir '%s'", self.workdir.name)
        self._get_generator()
        self.log.debug("Build generator with workdir '%s'", self.workdir.name)
        self._build_generator()

    def _get_generator(self):
        url = urllib.request.urlopen(self.URL)
        self.log.debug("Downloading '%s' from '%s'", self.NAME, self.URL)
        with ZipFile(BytesIO(url.read())) as my_zip_file:
            print(my_zip_file.__dict__)
            relevant_file = "cnfuzzdd2013/cnfuzz.c"
            self.log.debug("Writing source file to '%s'", self.sourcefile)
            with open((self.sourcefile), "wb") as output:
                for line in my_zip_file.open(relevant_file).readlines():
                    output.write(line)

    def _build_generator(self):
        build_call = ["gcc", "-O2", self.sourcefile,
                      "-o", self.generator]  # "-Wall"
        # TODO: forward output to log file, only display on non-zero staus
        self.log.debug("Building solver with %r", build_call)
        subprocess.call(build_call)

    def generate(self, formula, output_file):
        self.log.debug("Generate formula %d in file '%s' with generator '%s'",
                       formula, output_file, self.generator)
        generate_call = [self.generator, str(formula)]
        with open(output_file, "w") as outfile:
            self.log.debug("Calling generator '%r' with stdout='%s'",
                           generate_call, output_file)
            subprocess.run(generate_call, stdout=outfile)


class SATsolver(object):
    BINARY = ["build", "release", "bin", "mergesat"]
    NAME = "mergesat"
    REPO = "https://github.com/conp-solutions/mergesat.git"
    COMMIT = "306d2e8ef9733291acd6a07716c6158546a1c8d5"

    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.solver = None
        self.workdir = tempfile.TemporaryDirectory(
            prefix="specsat_solver", dir=os.getcwd())
        self.solverdir = os.path.join(self.workdir.name, self.NAME)
        self.log.debug("Run solver with workdir '%s'", self.workdir.name)

        self._get_solver(self.solverdir)
        self.log.info("Retrieved solver '%s' with version '%s'",
                      self.NAME, self._get_version())
        self._build_solver()
        assert self.solver != None

    def _get_solver(self, directory):
        clone_call = ["git", "clone", self.REPO, directory]
        self.log.debug("Cloning solver with: %r", clone_call)
        subprocess.call(clone_call)

    def _build_solver(self):
        build_call = ["make", "r", "-j", str(psutil.cpu_count())]
        self.log.debug("Building solver with: %r in cwd: '%s'",
                       build_call, self.solverdir)
        # TODO: forward output to log file, only display on non-zero staus
        subprocess.call(build_call, cwd=self.solverdir)
        self.solver = os.path.join(self.solverdir, *self.BINARY)

    def _get_version(self):
        version_call = ["git", "describe"]
        self.log.debug("Get solver version with: %r", version_call)
        process = subprocess.run(
            version_call, cwd=self.solverdir, capture_output=True)
        return process.stdout

    def solve_call(self, formula_path, cores):
        assert self.solver != None
        call = [self.solver, formula_path]
        self.log.debug("Generated solver call: '%r'", call)
        self.log.error("Ignoring number of cores '%d' for now")
        return call


class Benchmarker(object):
    BASE_WORK_DIR = "/dev/shm"  # For now, support Linux

    def __init__(self, **kwargs):
        self.log = logging.getLogger(self.__class__.__name__)
        self.__dict__.update(kwargs)

    def machine_details():
        all_cpus = psutil.cpu_count()
        cpus = psutil.cpu_count(logical=False)

    def run(self):
        old_cwd = os.getcwd()
        self.log.debug(
            "Starting Benchmarking Run in cwd '%s', changing to '%s'", old_cwd, self.BASE_WORK_DIR)
        os.chdir(self.BASE_WORK_DIR)

        self.log.debug("Get CNF Generator")
        generator = CNFgenerator()

        self.log.debug("Get SAT Solver")
        solver = SATsolver()

        # iterate over benchmark - short, medium, full settings

        # example round
        formula_path = os.path.join(self.BASE_WORK_DIR, "input.cnf")
        # TODO create meaningful list of benchmarks, add expected status code, conflicts and decisions, to check determinism
        benchmarks = [{
            "seed": 1,
            "base_cpu_time": 0.011,
            "expected_status": 10
        }, {
            "seed": 2,
            "base_cpu_time": 0.01,
        }, {
            "seed": 3,
            "base_cpu_time": 0.01,
        }
        ]

        # detect cores
        relevant_cores = [{"cores": 1, "name": "single"}]
        relevant_cores.append({"cores": psutil.cpu_count(
            logical=False), "name": "non-logical"})
        if psutil.cpu_count(logical=False) != psutil.cpu_count():
            relevant_cores.append(
                {"cores": psutil.cpu_count(), "name": "logical"})
        self.log.info("Detected cores: %r", relevant_cores)

        for benchmark in benchmarks:
            generator.generate(benchmark["seed"], formula_path)
            for core_data in relevant_cores:
                cores = core_data["cores"]
                solve_call = solver.solve_call(formula_path, cores)
                print(solve_call)
                # TODO: use actually useful call
                solve_result = measure_call(solve_call)
                solve_result["cores"] = core_data
                print(benchmark)
                print(solve_result)
        return 0


def parse_args():
    parser = argparse.ArgumentParser(description='Run SpecSAT')
    parser.add_argument('-d', '--debug', default=False,
                        action='store_true', help='Log debug output')
    parser.add_argument('-v', '--version', default=False,
                        action='store_true', help='Print version of the tool')

    args = parser.parse_args()
    return vars(args)


def main():
    args = parse_args()

    if args.pop("debug"):
        logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            datefmt='%Y-%m-%d:%H:%M:%S', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            datefmt='%Y-%m-%d:%H:%M:%S', level=logging.WARNING)

    if args.pop("version"):
        print("Version: {}".format(VERSION))
        return 0

    log.debug("Starting benchmarking with args: %r", args)
    benchmarker = Benchmarker(*args)
    return benchmarker.run()


if __name__ == "__main__":
    sys.exit(main())
