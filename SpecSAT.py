#!/usr/bin/python3

import argparse
import contextlib
import json
import logging
import os
import platform
import psutil
import requests  # downloading
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time


# Create logger
log = logging.getLogger(__name__)

VERSION = "0.0.1"  # Version of this tool


def get_thp_status():
    """Check host for THP status."""
    call = ["cat", "/sys/kernel/mm/transparent_hugepage/enabled"]
    try:
        log.debug("Looking up THP status with call %r", call)
        process = subprocess.run(call, capture_output=True)
        thp_status = str(process.stdout)
    except Exception as e:
        log.warning("Received exception %r when looking up THP status", e)
        return "unknown"
    output_candidates = thp_status.split(" ")
    for output in output_candidates:
        if output.startswith("["):
            return output.strip("[]")
    return "unknown"


def run_silently(call, **kwargs):
    """Run command, an donly print output in case of failure."""
    log.debug("Building solver with %r", call)
    process = subprocess.run(call, **kwargs, capture_output=True)
    if process.returncode != 0:
        log.error("Command %r failed with status %d", call, process.returncode)
        print("STDOUT: ", process.stdout.decode("utf_8"))
        print("STDERR: ", process.stderr.decode("utf_8"))
        raise Exception("Failed execution of %r %r", call, kwargs)


@contextlib.contextmanager
def pushd(new_dir):
    """Similar to shell's pushd, popd is implicit"""
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


def get_host_info():
    """Return a dictionary about the status of the host."""
    # CPUs (logical, physical, caches, model itendifier)
    # memory (size, type, model itendifier, TLB status)
    svmem = psutil.virtual_memory()
    return {
        "cpus_physical": psutil.cpu_count(),
        "cpus_logical": psutil.cpu_count(logical=False),
        "platform_system": platform.system(),
        "platform_processor": platform.processor(),
        "platform_version": platform.version(),
        "platform_release": platform.release(),
        "memory_total": svmem.total,
        "memory_available": svmem.available,
        "memory_thp_status": get_thp_status()
    }


def measure_call(call, output_file):
    """Run the given command, return measured performance data."""
    start_wallclock = time.perf_counter_ns()
    start_cpuclock = time.process_time_ns()
    # run command, ignore output
    # TODO extract status code of call
    process = subprocess.run(call, stdout=output_file,
                             stderr=subprocess.DEVNULL)
    end_wallclock = time.perf_counter_ns()
    end_cpuclock = time.process_time_ns()

    return {  # TODO: return status code of call
        "wall_time_ns": end_wallclock - start_wallclock,
        "cpu_time_ns": end_cpuclock - start_cpuclock,
        "status_code": process.returncode
    }


class CNFgenerator(object):

    NAME = "modgen"
    URL = "https://www.ugr.es/~jgiraldez/download/modularityGen_v2.1.tar.gz"
    VERSION = "modularityGen_v2.1"

    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info('creating an instance of Auxiliary')
        self.solver = None
        self.workdir = tempfile.TemporaryDirectory(
            prefix="specsat_solver", dir=os.getcwd())
        self.sourcefile = os.path.join(
            self.workdir.name, "modularityGen_v2.1.cpp")
        self.generator = os.path.join(self.workdir.name, self.NAME)
        self.log.debug("Get generator with workdir '%s'", self.workdir.name)
        self._get_generator()
        self.log.debug("Build generator with workdir '%s'", self.workdir.name)
        self._build_generator()

    def _get_generator(self):
        self.log.debug("Downloading '%s' from '%s'", self.NAME, self.URL)
        with pushd(self.workdir.name):
            targz_file_name = "modularityGen_v2.1.tar.gz"

            # download
            response = requests.get(self.URL, stream=True)
            with open(targz_file_name, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            del response

            # extract
            tar = tarfile.open(targz_file_name, mode="r:gz")
            tar.extractall()
            tar.close()
            print(os.getcwd())
            print(os.listdir())

    def _build_generator(self):
        build_call = ["g++", "-O2", self.sourcefile,
                      "-o", self.generator]  # "-Wall"
        # TODO: forward output to log file, only display on non-zero staus
        self.log.debug("Building solver with %r", build_call)
        run_silently(build_call)

    def generate(self, output_file, parameter=None):
        self.log.debug("Generate formula in file '%s' with parameters %r",
                       output_file, parameter)
        generate_call = [self.generator]
        if parameter is not None:
            generate_call += parameter
        with open(output_file, "w") as outfile:
            self.log.debug("Calling generator '%r' with stdout='%s'",
                           generate_call, output_file)
            subprocess.run(generate_call, stdout=outfile)

    def get_name(self):
        return self.NAME

    def get_version(self):
        return VERSION


class SATsolver(object):
    BINARY = ["build", "release", "bin", "mergesat"]
    NAME = "mergesat"
    REPO = "https://github.com/conp-solutions/mergesat.git"
    COMMIT = "306d2e8ef9733291acd6a07716c6158546a1c8d5"
    SOLVER_PARAMETER = ["-no-diversify"]

    def __init__(self, compiler=None, compile_flags=None):
        self.log = logging.getLogger(self.__class__.__name__)
        self.build_command = None
        self.solver = None
        self.version = None
        self.workdir = tempfile.TemporaryDirectory(
            prefix="specsat_solver", dir=os.getcwd())
        self.solverdir = os.path.join(self.workdir.name, self.NAME)
        self.log.debug("Run solver with workdir '%s'", self.workdir.name)

        self._get_solver(self.solverdir)
        self.log.info("Retrieved solver '%s' with version '%s'",
                      self.NAME, self._get_version())
        self._build_solver(compiler=compiler, compile_flags=compile_flags)
        assert self.solver != None
        assert self.build_command != None

    def _get_solver(self, directory):
        clone_call = ["git", "clone", self.REPO, directory]
        self.log.debug("Cloning solver with: %r", clone_call)
        run_silently(clone_call)

    def _build_solver(self, compiler=None, compile_flags=None):
        self.build_command = ["make", "BUILD_TYPE=parallel",
                              "r", "-j", str(psutil.cpu_count())]
        if compiler is not None:
            self.build_command.append(f"CXX={compiler}")
        if compile_flags is not None:
            self.build_command.append(f"CXX_EXTRA_FLAGS={compile_flags}")
            self.build_command.append(f"LD_EXTRA_FLAGS={compile_flags}")
        self.log.debug("Building solver with: %r in cwd: '%s'",
                       self.build_command, self.solverdir)
        # TODO: forward output to log file, only display on non-zero staus
        run_silently(self.build_command, cwd=self.solverdir)
        # subprocess.call(self.build_command, cwd=self.solverdir)
        self.solver = os.path.join(self.solverdir, *self.BINARY)

    def _get_version(self):
        if self.version is None:
            version_call = ["git", "describe"]
            self.log.debug("Get solver version with: %r", version_call)
            process = subprocess.run(
                version_call, cwd=self.solverdir, capture_output=True)
            self.version = process.stdout.strip().decode('utf-8')
        return self.version

    def solve_call(self, formula_path, cores):
        assert self.solver != None
        call = [self.solver] + self.SOLVER_PARAMETER + \
            [f"-cores={cores}"] + [formula_path]
        self.log.debug("Generated solver call: '%r'", call)
        return call

    def get_build_command(self):
        return self.build_command

    def get_name(self):
        return self.NAME

    def get_version(self):
        return self._get_version()


class Benchmarker(object):
    BASE_WORK_DIR = "/dev/shm"  # For now, support Linux

    def __init__(self, solver, generator, **kwargs):
        self.log = logging.getLogger(self.__class__.__name__)
        self.__dict__.update(kwargs)
        self.log.debug("Get SAT Solver")
        self.solver = solver
        self.generator = generator
        self.fail_early = False  # TODO: make this a parameter that is updated in the line above

    def run(self):
        old_cwd = os.getcwd()
        self.log.debug(
            "Starting Benchmarking Run in cwd '%s', changing to '%s'", old_cwd, self.BASE_WORK_DIR)
        os.chdir(self.BASE_WORK_DIR)

        report = {}
        report["raw_runs"] = []
        report["failed_runs"] = 0

        self.log.debug("Get CNF Generator")

        report["generator"] = {
            "name": self.generator.get_name(),
            "version": self.generator.get_version()
        }

        report["satsolver"] = {
            "name": self.solver.get_name(),
            "version": self.solver.get_version(),
            "build_command": self.solver.get_build_command()
        }

        report["hostinfo"] = get_host_info()

        # detect cores
        relevant_cores = [{"cores": 1, "name": "single"}]
        relevant_cores.append({"cores": psutil.cpu_count(
            logical=False), "name": "physical_cores"})
        if psutil.cpu_count(logical=False) != psutil.cpu_count():
            relevant_cores.append({"cores": psutil.cpu_count(
                logical=False), "name": "logical_cores"})

        formula_path = os.path.join(self.BASE_WORK_DIR, "input.cnf")
        output_path = os.path.join(self.BASE_WORK_DIR, "output.log")

        # TODO add unsat formulas to the list
        benchmarks = [{
            "parameter": ["-s", "4900", "-n", "1000000", "-m", "3000000"],
            "base_cpu_time": 25,
            "expected_status": 10
        }, {
            "parameter": ["-s", "10000000", "-n", "100000", "-m", "340000"],
            "base_cpu_time": 240,
            "expected_status": 10
        }, {
            "parameter": ["-s", "3900", "-n", "10000", "-m", "38000"],
            "base_cpu_time": 35,
            "expected_status": 10
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

        detected_failure = False
        for benchmark in benchmarks:
            if detected_failure and self.fail_early:
                self.log.warning("Stopping execution due to detected error")
                break

            self.generator.generate(formula_path, benchmark["parameter"])
            for core_data in relevant_cores:
                if detected_failure and self.fail_early:
                    break
                okay_run = True
                cores = core_data["cores"]
                solve_call = self.solver.solve_call(formula_path, cores)
                print(solve_call)
                # TODO: use actually useful call
                with open(output_path, "w") as output_file:
                    solve_result = measure_call(solve_call, output_file)
                solve_result["cores"] = core_data
                solve_result["call"] = solve_call
                log.debug("Solved formula %r with '%r'",
                          benchmark["parameter"], solve_result)
                if solve_result["status_code"] != benchmark["expected_status"]:
                    self.log.error("failed formula %r with unmatching status code '%d' instead of expected '%d'",
                                   benchmark["parameter"], solve_result["status_code"], benchmark["expected_status"])
                    detected_failure = True
                    okay_run = False
                    report["failed_runs"] += 1
                solve_result["okay"] = okay_run
                solve_result["benchmark"] = benchmark
                # TODO: also compare expected decisions and expected conflicts
                report["raw_runs"].append(solve_result)
                print(benchmark)
                print(solve_result)

        # Print report (and store if output file is specified)
        specsat_report = {}
        specsat_report["SpecSAT"] = report
        self.log.info("Printing report %r", specsat_report)
        print(json.dumps(specsat_report, indent=4, sort_keys=True))

        return 0


def parse_args():
    parser = argparse.ArgumentParser(description='Run SpecSAT')
    parser.add_argument('-d', '--debug', default=False,
                        action='store_true', help='Log debug output')
    parser.add_argument('-v', '--version', default=False,
                        action='store_true', help='Print version of the tool')

    parser.add_argument('--sat-compiler', default=None,
                        help='Use this compiler as CXX')
    parser.add_argument('--sat-compile-flags', default=None,
                        help='Add this string to CXXFLAGS and LDFLAGS')

    args = parser.parse_args()
    return vars(args)


def main():
    args = parse_args()

    if args.pop("debug"):
        logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            datefmt='%Y-%m-%d:%H:%M:%S', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)

    if args.pop("version"):
        print("Version: {}".format(VERSION))
        return 0

    log.info("Building CNF generator")
    generator = CNFgenerator()
    log.debug("Build generator '%s' with version '%s'",
              generator.get_name(), generator.get_version())

    log.debug("Pre-SAT args: %r", args)
    log.info("Building SAT solver")
    sat_args = ["sat_compiler", "sat_compile_flags"]
    satsolver = SATsolver(compiler=args.get("sat_compiler"),
                          compile_flags=args.get("sat_compile_flags"))
    for sat_arg in sat_args:
        if sat_arg in args:
            args.pop(sat_arg)

    log.debug("Starting benchmarking with args: %r", args)
    benchmarker = Benchmarker(solver=satsolver, generator=generator, *args)
    return benchmarker.run()


if __name__ == "__main__":
    sys.exit(main())
