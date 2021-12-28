#!/usr/bin/python3

import argparse
import contextlib
import copy
import cpuinfo
import datetime
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

from collections import defaultdict
from statistics import variance

# Create logger
log = logging.getLogger(__name__)

VERSION = "0.0.1"  # Version of this tool


def get_thp_status():
    """Check host for THP status."""
    call = ["cat", "/sys/kernel/mm/transparent_hugepage/enabled"]
    try:
        log.debug("Looking up THP status with call %r", call)
        process = subprocess.run(call, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
    log.debug("Silently executing call %r", call)
    process = subprocess.run(
        call, **kwargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
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
        "cpu_info": cpuinfo.get_cpu_info(),
        "platform_system": platform.system(),
        "platform": platform.platform(),
        "platform_processor": platform.processor(),
        "platform_version": platform.version(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "memory_total": svmem.total,
        "memory_available": svmem.available,
        "memory_thp_status": get_thp_status(),
    }


def measure_call(call, output_file):
    """Run the given command, return measured performance data."""
    pre_stats = os.times()
    process = subprocess.run(call, stdout=output_file, stderr=subprocess.DEVNULL)
    post_stats = os.times()
    return {
        "cpu_time_s": (post_stats[2] + post_stats[3]) - (pre_stats[2] + pre_stats[3]),
        "wall_time_s": post_stats[4] - pre_stats[4],
        "status_code": process.returncode,
    }


class CNFgenerator(object):

    NAME = "modgen"
    URL = "https://www.ugr.es/~jgiraldez/download/modularityGen_v2.1.tar.gz"
    VERSION = "modularityGen_v2.1"

    def __init__(self, cxx="g++", tool_location=None):
        self.log = logging.getLogger(self.__class__.__name__)
        self.solver = None
        self.workdir = tempfile.TemporaryDirectory(
            prefix="specsat_solver", dir=os.getcwd()
        )
        self.sourcefile = os.path.join(self.workdir.name, "modularityGen_v2.1.cpp")
        if tool_location is None:
            self.generator = os.path.join(self.workdir.name, self.NAME)
            self.log.debug("Get generator with workdir '%s'", self.workdir.name)
            self._get_generator()
            self.log.debug("Build generator with workdir '%s'", self.workdir.name)
            self._build_generator(cxx=cxx)
        else:
            if not os.path.isfile(tool_location) and os.access(tool_location, os.X_OK):
                error = f"Provided generator tool '{tool_location}' is not executable, aborting"
                log.error(error)
                raise ValueError(error)
            log.info("Using user provided modgen tool, results might differ!")
            self.generator = os.path.realpath(tool_location)
            self.VERSION = "<UserProvided>"

    def _get_generator(self):
        self.log.debug("Downloading '%s' from '%s'", self.NAME, self.URL)
        with pushd(self.workdir.name):
            targz_file_name = "modularityGen_v2.1.tar.gz"

            # download
            response = requests.get(self.URL, stream=True)
            with open(targz_file_name, "wb") as out_file:
                shutil.copyfileobj(response.raw, out_file)
            del response

            # extract
            tar = tarfile.open(targz_file_name, mode="r:gz")
            tar.extractall()
            tar.close()

    def _build_generator(self, cxx="g++"):
        build_call = [cxx, "-O2", self.sourcefile, "-o", self.generator]  # "-Wall"
        self.log.debug("Building solver with %r", build_call)
        run_silently(build_call)

    def generate(self, output_file, parameter=None):
        self.log.debug(
            "Generate formula in file '%s' with parameters %r", output_file, parameter
        )
        generate_call = [self.generator]
        if parameter is not None:
            generate_call += parameter
        with open(output_file, "w") as outfile:
            self.log.debug(
                "Calling generator '%r' with stdout='%s'", generate_call, output_file
            )
            subprocess.run(generate_call, stdout=outfile)

    def get_name(self):
        return self.NAME

    def get_version(self):
        return self.VERSION


class SATsolver(object):
    NAME = "mergesat"
    REPO = "https://github.com/conp-solutions/mergesat.git"
    SOLVER_PARAMETER = ["-no-diversify"]

    def __init__(
        self,
        compiler=None,
        compile_flags=None,
        commit=None,
        mode=None,
        solver_location=None,
    ):
        self.log = logging.getLogger(self.__class__.__name__)
        self.build_command = None
        self.mode = "debug" if mode == "debug" else "release"
        self.solver = None
        self.version = None
        self.workdir = tempfile.TemporaryDirectory(
            prefix="specsat_solver", dir=os.getcwd()
        )

        if solver_location is None:
            self.solverdir = os.path.join(self.workdir.name, self.NAME)
            self.log.debug("Run solver with workdir '%s'", self.workdir.name)
            self._get_solver(self.solverdir, commit=commit)
            self.log.info(
                "Retrieved solver '%s' with version '%s'",
                self.NAME,
                self._get_version(),
            )
            self.binary = ["build", self.mode, "bin", "mergesat"]
            self._build_solver(compiler=compiler, compile_flags=compile_flags)
            assert self.build_command != None
        else:
            self.solverdir = None
            if not os.path.isfile(solver_location) and os.access(
                solver_location, os.X_OK
            ):
                error = f"Provided SAT solver '{solver_location}' is not executable, aborting"
                log.error(error)
                raise ValueError(error)
            log.info("Using user provided SAT solver, results might differ!")
            self.solver = os.path.realpath(solver_location)
        assert self.solver != None

    def _get_solver(self, directory, commit=None):
        self.log.debug("get solver: %r", locals())
        clone_call = ["git", "clone", self.REPO, directory]
        self.log.debug("Cloning solver with: %r", clone_call)
        run_silently(clone_call)
        if commit:
            checkout_call = ["git", "reset", "--hard", commit]
            self.log.debug("Select SAT commit %r", checkout_call)
            with pushd(directory):
                run_silently(checkout_call)

    def _build_solver(self, compiler=None, compile_flags=None):
        self.build_command = [
            "make",
            "BUILD_TYPE=parallel",
            "d" if self.mode == "debug" else "r",
            "-j",
            str(psutil.cpu_count()),
        ]
        if compiler is not None:
            self.build_command.append(f"CXX={compiler}")
        if compile_flags is not None:
            self.build_command.append(f"CXX_EXTRA_FLAGS={compile_flags}")
            self.build_command.append(f"LD_EXTRA_FLAGS={compile_flags}")
        self.log.debug(
            "Building solver with: %r in cwd: '%s'", self.build_command, self.solverdir
        )
        run_silently(self.build_command, cwd=self.solverdir)
        self.solver = os.path.join(self.solverdir, *self.binary)

    def _get_version(self):
        if self.solverdir is None:
            return "<UserProvided>"
        if self.version is None:
            version_call = ["git", "describe"]
            self.log.debug("Get solver version with: %r", version_call)
            process = subprocess.run(
                version_call,
                cwd=self.solverdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.version = process.stdout.strip().decode("utf-8")
        return self.version

    def solve_call(self, formula_path, cores):
        assert self.solver != None
        call = [self.solver] + self.SOLVER_PARAMETER + [f"-cores={cores}"]
        if cores > 1:
            call += ["-no-pre"]
        call += [formula_path]
        self.log.debug("Generated solver call: '%r'", call)
        return call

    def get_build_command(self):
        return self.build_command

    def get_name(self):
        return self.NAME

    def get_version(self):
        return self._get_version()

    def validate_conflicts(self, expected_conflicts, cores, log_file_name):
        """Check whether for given cores, conflicts have been detected."""

        match_line = "c SUM stats conflicts:           :"
        # parallel solvers report SUM of all conflicts, hence, multiply
        match_conflicts = int(expected_conflicts) * cores

        with open(log_file_name) as log_file:
            all_lines = log_file.readlines()
            for line in all_lines:
                if line.startswith(match_line):
                    # extract conflicts
                    log.debug("Extracting conflicts from line '%s'", line)
                    conflicts = int(line.split(":")[2])
                    log.debug("Extraced %d conflicts with %d cores", conflicts, cores)
                    if match_conflicts == conflicts:
                        return True
                    else:
                        self.log.warning(
                            "Expected conflicts %d for cores %d do not match detected conflicts %d - please report mismatch to author of SpecSAT and MergeSat",
                            match_conflicts,
                            cores,
                            conflicts,
                        )
                        return False

        # In case we fail to match anything successfully, fail overall
        return False


class Benchmarker(object):
    BASE_WORK_DIR = "/dev/shm"  # For now, support Linux

    def __init__(self, solver, generator, used_user_tools):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.debug("Get SAT Solver")
        self.solver = solver
        self.generator = generator
        self.relevant_cores = None
        self.fail_early = False
        self.used_user_tools = used_user_tools

    def _prepare_report(self):
        report = {}
        report["raw_runs"] = []
        report["failed_runs"] = 0

        self.log.debug("Preparing report with tool and host info")

        report["generator"] = {
            "name": self.generator.get_name(),
            "version": self.generator.get_version(),
        }

        report["satsolver"] = {
            "name": self.solver.get_name(),
            "version": self.solver.get_version(),
            "build_command": self.solver.get_build_command(),
        }
        report["hostinfo"] = get_host_info()
        report["relevant_cores"] = self._detect_cores()
        report["non_default_tools"] = self.used_user_tools
        return report

    def _get_benchmarks(self, only_one=False):
        # benchmark.get("expected_sequential_conflicts" if cores == 1 else "expected_1parallel_conflicts")
        benchmarks = [
            {
                "parameter": ["-s", "4900", "-n", "1000", "-m", "3000"],
                "base_sequential_cpu_time": 25,
                "expected_sequential_conflicts": 48,
                "expected_status": 10,
            },
            {
                "parameter": ["-s", "2400", "-n", "15000", "-m", "72500"],
                "base_sequential_cpu_time": 20,
                "expected_sequential_conflicts": 728584,
                "expected_status": 20,
            },
            {
                "parameter": ["-s", "4900", "-n", "1000000", "-m", "3000000"],
                "base_sequential_cpu_time": 25,
                "expected_sequential_conflicts": 352,
                "expected_status": 10,
            },
            {
                "parameter": ["-s", "3900", "-n", "10000", "-m", "38000"],
                "base_sequential_cpu_time": 35,
                "expected_sequential_conflicts": 606635,
                "expected_status": 10,
            },
            {
                "parameter": ["-n", "2200", "-m", "9086", "-c", "40", "-s", "158"],
                "base_sequential_cpu_time": 100,
                "expected_sequential_conflicts": 594464,
                "expected_status": 10,
            },
            {
                "parameter": ["-n", "45000", "-m", "171000", "-c", "40", "-s", "100"],
                "base_sequential_cpu_time": 100,
                "expected_sequential_conflicts": 2172508,
                "expected_status": 10,
                "restriction": "sequential",
            },
            {
                "parameter": ["-n", "52500", "-m", "194250", "-c", "40", "-s", "100"],
                "base_sequential_cpu_time": 100,
                "expected_status": 10,
                "restriction": "parallel",
            },
        ]
        return benchmarks if not only_one else [benchmarks[0]]

    def _detect_cores(self):
        if self.relevant_cores:
            return self.relevant_cores
        self.relevant_cores = [{"cores": 1, "name": "single"}]
        self.relevant_cores.append(
            {"cores": psutil.cpu_count(logical=False), "name": "non-logical"}
        )
        if psutil.cpu_count(logical=False) != psutil.cpu_count():
            self.relevant_cores.append({"cores": psutil.cpu_count(), "name": "logical"})
        half_cores = psutil.cpu_count(logical=False) // 2
        if half_cores != 1:
            self.relevant_cores.append({"cores": half_cores, "name": "half-cores"})
        self.log.info("Detected cores: %r", self.relevant_cores)
        return self.relevant_cores

    def _run_iterations(self, report, iteration, lite=False):
        with pushd(self.BASE_WORK_DIR):
            benchmarks = self._get_benchmarks(only_one=lite)
            relevant_cores = self._detect_cores()

            formula_path = os.path.join(self.BASE_WORK_DIR, "input.cnf")
            output_path = os.path.join(self.BASE_WORK_DIR, "output.log")

            detected_failure = False
            for benchmark in benchmarks:
                if detected_failure and self.fail_early:
                    self.log.warning("Stopping execution due to detected error")
                    break

                log.info("Solving benchmark %r", benchmark)
                self.generator.generate(formula_path, benchmark["parameter"])
                for core_data in relevant_cores:
                    if detected_failure and self.fail_early:
                        break
                    restriction = benchmark.get("restriction", "")
                    cores = core_data["cores"]
                    # Check whether we should run here!
                    if (restriction == "sequential" and cores != 1) or (
                        restriction == "parallel" and cores == 1
                    ):
                        log.debug(
                            "Skip benchmark restricted to %s with %d cores",
                            restriction,
                            cores,
                        )
                        continue
                    okay_run = True
                    solve_call = self.solver.solve_call(formula_path, cores)
                    log.debug(
                        "Solving formula %r and cores %d with solving call %r",
                        benchmark,
                        cores,
                        solve_call,
                    )

                    with open(output_path, "w") as output_file:
                        solve_result = measure_call(solve_call, output_file)
                    if cores == 1:
                        if not self.solver.validate_conflicts(
                            benchmark.get("expected_sequential_conflicts"),
                            cores,
                            output_path,
                        ):
                            detected_failure = True
                            solve_result["validated"] = False
                        else:
                            solve_result["validated"] = None
                    solve_result["iteration"] = iteration
                    solve_result["cores"] = core_data
                    solve_result["call"] = solve_call
                    log.debug(
                        "Solved formula %r with '%r'",
                        benchmark["parameter"],
                        solve_result,
                    )
                    if solve_result["status_code"] != benchmark["expected_status"]:
                        self.log.error(
                            "failed formula %r with unmatching status code '%d' instead of expected '%d'",
                            benchmark["parameter"],
                            solve_result["status_code"],
                            benchmark["expected_status"],
                        )
                        detected_failure = True
                        okay_run = False
                        report["failed_runs"] += 1
                    solve_result["okay"] = okay_run
                    solve_result["benchmark"] = benchmark
                    report["raw_runs"].append(solve_result)
                    log.debug(
                        "For formula %r and cores %d, obtained results %r",
                        benchmark,
                        cores,
                        solve_result,
                    )
        return detected_failure

    def _generate_summary(self, report):

        self.log.info("Generating Summary")
        sum_sequential_wall = 0
        num_sequential_runs = 0
        sum_max_parallel_wall = 0
        sum_max_parallel_efficiency = 0
        num_max_parallel_runs = 0
        parallel_stats = {}
        max_cores = 1
        relevant_cores = self._detect_cores()
        for cores in relevant_cores:
            parallel_stats[cores["cores"]] = {
                "sum_max_parallel_wall": 0,
                "sum_max_parallel_efficiency": 0,
                "num_max_parallel_runs": 0,
            }
        for run in report["raw_runs"]:
            cores = run["cores"]["cores"]
            max_cores = cores if cores > max_cores else max_cores
        self.log.debug("Detected max cores %r", max_cores)

        iteration_wall_runtimes = defaultdict(list)  # Collect all runtimes per run
        iteration_cpu_runtimes = defaultdict(list)  # Collect all runtimes per run
        self.log.debug("Evaluating %d runs", len(report["raw_runs"]))
        for run in report["raw_runs"]:
            cores = run["cores"]["cores"]
            self.log.debug("Using run with %r cores", cores)
            if cores == 1:
                sum_sequential_wall += run["wall_time_s"]
                num_sequential_runs += 1
            elif cores == max_cores:
                sum_max_parallel_wall += run["wall_time_s"]
                sum_max_parallel_efficiency += (
                    run["cpu_time_s"] / (max_cores * run["wall_time_s"])
                    if run["wall_time_s"]
                    else 1
                )
                num_max_parallel_runs += 1
            iteration_key = (tuple(run["benchmark"]["parameter"]), cores)
            iteration_cpu_runtimes[iteration_key].append(run["cpu_time_s"])
            iteration_wall_runtimes[iteration_key].append(run["wall_time_s"])

            parallel_stats[cores]["sum_max_parallel_wall"] += run["wall_time_s"]
            parallel_stats[cores]["sum_max_parallel_efficiency"] += (
                run["cpu_time_s"] / (max_cores * run["wall_time_s"])
                if max_cores * run["wall_time_s"]
                else 1
            )
            parallel_stats[cores]["num_max_parallel_runs"] += 1

        # Get variance based on run with this highest sum of run times
        max_iterations = 0
        cpu_list = None
        cpu_sum = 0
        for item in iteration_cpu_runtimes.values():
            max_iterations = max_iterations if max_iterations > len(item) else len(item)
            item_sum = sum(item)
            if cpu_list is None or item_sum > cpu_sum:
                cpu_list = item
                cpu_sum = item_sum
        wall_list = None
        wall_sum = 0
        for item in iteration_cpu_runtimes.values():
            item_sum = sum(item)
            if wall_list is None or item_sum > wall_sum:
                wall_list = item
                wall_sum = item_sum
        log.info("Detected %d iterations", max_iterations)
        cpu_variance = 0 if len(cpu_list) <= 1 else variance(cpu_list)
        wall_variance = 0 if len(wall_list) <= 1 else variance(wall_list)

        log.debug(
            "Detected parallel values: sum_efficiency: %r parallel runs: %r max_cores: %r",
            sum_max_parallel_efficiency,
            num_max_parallel_runs,
            max_cores,
        )
        log.debug(
            "Detected machine variance: %f cpu time, %f wall time",
            cpu_variance,
            wall_variance,
        )
        # Plain wait time to result, average via runs, so that multiple iterations still result in same score
        sequential_score = sum_sequential_wall / num_sequential_runs * 100
        # Wait time to result, 1 result per core, average via runs so that multiple iterations still result in same score
        parallel_score = (
            sum_max_parallel_wall / (max_cores * num_max_parallel_runs) * 100
        )
        # With higher efficiency per core, we get better. Hence, use efficiency to limit factor.
        # TODO: instead of (2-x), should this be (1/x) ?
        efficiency_score = (
            sum_max_parallel_wall
            / (num_max_parallel_runs * max_cores)
            * (2 - (sum_max_parallel_efficiency / num_max_parallel_runs))
        )
        # TODO: evaluate efficiency between highest three core numbers, take 'logical' into account
        log.debug(
            "Detected scores: sequential: %r parallel: %r efficiency: %r",
            sequential_score,
            parallel_score,
            efficiency_score,
        )

        report["summary"] = {
            "total_runs": len(report["raw_runs"]),
            "score_sequential": sequential_score,
            "score_full_parallel": parallel_score,
            "score_efficiency": efficiency_score,
            "wall_time_sum_seq_s": sum_sequential_wall,
            "wall_time_sum_par_s": sum_max_parallel_wall,
            "wall_time_variance": wall_variance,
            "cpu_time_variance": cpu_variance,
            "measurement_iterations": max_iterations,
            "efficiency_max_parallel_avg": sum_max_parallel_efficiency
            / num_max_parallel_runs
            if num_max_parallel_runs
            else 0,
            "detailed_stats": parallel_stats,
        }

        # Print Score
        if self.used_user_tools:
            print("!!!  Attention: non-default tools have been used  !!!")
        print("Sequential Score:    {} (less is better)".format(sequential_score))
        print("Full Parallel Score: {} (less is better)".format(parallel_score))
        print("Efficiency Score:    {} (less is better)".format(efficiency_score))
        print("Variance CPU time:   {} (less is better)".format(cpu_variance))
        print("Variance wall time:  {} (less is better)".format(wall_variance))

    def run(self, iterations=1, lite=False, verbosity=0):
        old_cwd = os.getcwd()

        if iterations is None or iterations < 1:
            self.log.warning(
                "Detected invalid value for iterations '%r', replacing with 1."
            )
            iterations = 1
        self.log.debug(
            "Starting Benchmarking Run in cwd '%s', changing to '%s'",
            old_cwd,
            self.BASE_WORK_DIR,
        )

        report = self._prepare_report()
        # Add all iterations to report
        report["start"] = datetime.datetime.now().isoformat()
        report["lite_variant"] = lite
        detected_failure = False
        for iteration in range(1, iterations + 1):
            if self._run_iterations(report, iteration=iteration, lite=lite):
                log.warning("Detected a failure in iteration %d", iteration)
                detected_failure = True
        report["end"] = datetime.datetime.now().isoformat()
        report["detected_failure"] = detected_failure

        self._generate_summary(report)
        log.debug("Showed report summary: %r", report["summary"])

        # Assemble report
        specsat_report = {}
        specsat_report["SpecSAT"] = report

        if verbosity > 0:
            self.log.debug("Printing report %r", specsat_report)
            print(json.dumps(specsat_report, indent=4, sort_keys=True))

        if detected_failure:
            log.error("Detected unexpected behavior")
        return specsat_report


def parse_args():
    parser = argparse.ArgumentParser(description="Run SpecSAT")
    parser.add_argument(
        "-d", "--debug", default=False, action="store_true", help="Log debug output"
    )
    parser.add_argument(
        "-l",
        "--lite",
        default=False,
        action="store_true",
        help="Only run a single, easy, benchmark to test the setup",
    )
    parser.add_argument(
        "-i", "--iterations", default=1, type=int, help="Re-run a run multiple times"
    )
    parser.add_argument(
        "-n",
        "--nick-name",
        default=None,
        help="Add this name as nick-name to the report.",
    )
    parser.add_argument(
        "-o", "--output", default=None, help="Write output to this file."
    )
    parser.add_argument(
        "-r",
        "--report",
        default=None,
        help="Write full report to this file, including raw data per benchmark.",
    )
    parser.add_argument(
        "-v",
        "--version",
        default=False,
        action="store_true",
        help="Print version of the tool",
    )
    parser.add_argument(
        "--verbosity", default=0, type=int, help="Set the verbosity level"
    )
    parser.add_argument(
        "--work-dir",
        default="/dev/shm",
        type=str,
        help="Build and run tools in this directory",
    )

    parser.add_argument(
        "--generator-cxx",
        default="g++",
        help="Use this compiler as CXX to compile the generator",
    )
    parser.add_argument(
        "-G",
        "--generator-location",
        default=None,
        type=str,
        help="Location of modgen tool to be used",
    )

    parser.add_argument(
        "--sat-commit", default="v3.2.0", help="Use this commit of the SAT solver"
    )
    parser.add_argument("--sat-compiler", default=None, help="Use this compiler as CXX")
    parser.add_argument(
        "--sat-compile-flags",
        default=None,
        help="Add this string to CXXFLAGS and LDFLAGS",
    )
    parser.add_argument(
        "--sat-mode",
        default="release",
        choices=["release", "debug"],
        help="Use solver in release or debug mode",
    )
    parser.add_argument(
        "-S",
        "--solver-location",
        default=None,
        type=str,
        help="Location of SAT solver to be used",
    )

    args = parser.parse_args()
    return vars(args)


def write_report(report, args):
    """Write output files based on args."""
    nick_name = args.get("nick_name")
    output_report = copy.deepcopy(report)
    output_report["SpecSAT"]["cli_args"] = args
    if nick_name:
        output_report["nick_name"] = nick_name
    report_file = args.get("report")
    if report_file:
        with open(report_file, "w") as f:
            json.dump(output_report, f, indent=4, sort_keys=True)
    output_file = args.get("output")
    if output_file:
        output_report["SpecSAT"].pop("raw_runs")
        with open(output_file, "w") as f:
            json.dump(report, f, indent=4, sort_keys=True)


def build_tools(args):
    """Build required tools, report whether user provided tools have been used."""
    generator_location = args.get("generator_location")
    generator_location = (
        generator_location
        if generator_location is None
        else os.path.realpath(generator_location)
    )
    solver_location = args.get("solver_location")
    solver_location = (
        solver_location
        if solver_location is None
        else os.path.realpath(solver_location)
    )
    with pushd(args["work_dir"]):
        log.info("Building CNF generator")
        generator = CNFgenerator(
            cxx=args.get("generator_cxx"), tool_location=generator_location
        )
        log.debug(
            "Build generator '%s' with version '%s'",
            generator.get_name(),
            generator.get_version(),
        )

        log.debug("Pre-SAT args: %r", args)
        log.info("Building SAT solver")
        satsolver = SATsolver(
            compiler=args.get("sat_compiler"),
            compile_flags=args.get("sat_compile_flags"),
            commit=args.get("sat_commit"),
            mode=args.get("sat_mode"),
            solver_location=solver_location,
        )
    return (
        satsolver,
        generator,
        generator_location is not None or solver_location is not None,
    )


def main():
    args = parse_args()

    if args.get("debug"):
        logging.basicConfig(
            format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d:%H:%M:%S",
            level=logging.DEBUG,
        )
    else:
        logging.basicConfig(
            format="%(asctime)s,%(msecs)d %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d:%H:%M:%S",
            level=logging.INFO,
        )

    if args.get("version"):
        print("Version: {}".format(VERSION))
        return 0

    log.info("SpecSAT 2021, version %s", VERSION)

    satsolver, generator, used_user_tools = build_tools(args)

    log.debug("Starting benchmarking with args: %r", args)
    benchmarker = Benchmarker(
        solver=satsolver, generator=generator, used_user_tools=used_user_tools
    )
    report = benchmarker.run(
        iterations=args.get("iterations"),
        lite=args.get("lite", False),
        verbosity=args.get("verbosity"),
    )
    write_report(report, args)
    log.info("Finished SpecSAT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
