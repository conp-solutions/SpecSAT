#!/usr/bin/env python3
#
# Copyright (C) 2021, Norbert Manthey <nmanthey@conp-solutions.com>

import argparse
import contextlib
import copy
import cpuinfo
import datetime
import glob
import hashlib
import json
import logging
import os
import platform
import psutil
import requests  # downloading
import resource  # process limits
import shutil
import subprocess
import sys
import tarfile
import tempfile

from collections import defaultdict
from statistics import variance

# Create logger
log = logging.getLogger(__name__)

DOCKER_NETWORK = []
FAST_WORK_DIR = tempfile.TemporaryDirectory(
    dir=None if sys.platform == "darwin" else "/dev/shm"
)
FAST_WORK_DIR_NAME = FAST_WORK_DIR.name
VERSION = "0.2.0"  # Version of this tool
ZIP_BSAE_DIRNAME = "SpecSAT_data"


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


def build_docker_container(dockerfile_dir=None):
    """Build docker container, and return hash."""
    log.info("Building docker container")
    dockerfile_dir = (
        dockerfile_dir
        if dockerfile_dir is not None
        else os.path.dirname(os.path.realpath(__file__))
    )
    cmd = ["docker", "build"] + DOCKER_NETWORK + ["-q", "."]
    cwd = dockerfile_dir
    log.debug("Using command %r to build docker container, in %s", cmd, cwd)
    process = subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if process.returncode != 0:
        log.error("Command %r failed with status %d", cmd, process.returncode)
        print("STDOUT: ", process.stdout.decode("utf_8"))
        print("STDERR: ", process.stderr.decode("utf_8"))
        raise Exception("Failed execution of %r", cmd)
    return process.stdout.strip().decode("utf-8")


def construct_docker_run_command(container_id, work_dir=None, extra_env=None):
    """Create cmd to jail with docker, an only mount work dir and /dev/shm."""
    base_id = container_id.split(":")[1]
    log.debug("Docker command with work_dir '%s'", work_dir)
    work_dir = work_dir if work_dir else os.getcwd()
    env_list = []
    if extra_env is not None:
        for e in extra_env:
            env_list += ["-e", e]
    run_command = (
        ["docker", "run"]
        + DOCKER_NETWORK
        + ["--rm", "-e", f'USER="{os.getlogin()}']
        + env_list
        + [
            f"-u={os.getuid()}",
            "-v",
            f"{work_dir}:{work_dir}",
            "-v",
            "/dev/shm:/dev/shm",
            "-w",
            f"{work_dir}",
            base_id,
        ]
    )
    return run_command


def get_container_call(container_id, call, extra_env=None, **kwargs):
    """Create full cmd, consider kwargs content wrt cwd."""
    if container_id is None:
        full_call = call
    else:
        # Jail with container, if requested
        cmd_prefix = construct_docker_run_command(
            container_id, work_dir=kwargs.get("cwd", os.getcwd()), extra_env=extra_env
        )
        full_call = cmd_prefix + call
        log.debug("Prepare full call %r", full_call)

    log.debug(
        "Generated full call %r from call %r with args %r with cwd '%s'",
        full_call,
        call,
        kwargs,
        os.getcwd(),
    )
    return full_call


def set_hour_timeout():
    """Set max runtime of the calling process to 3600 seconds."""
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))


def set_5m_timeout():
    """Set max runtime of the calling process to 300 seconds."""
    resource.setrlimit(resource.RLIMIT_CPU, (300, 300))


def get_env_with_extra(extra_env=None):
    runenv = dict(os.environ)
    if extra_env is not None:
        for item in extra_env:
            log.debug("Add/update env item %r", item)
            runenv[item.split("=")[0]] = "=".join(item.split("=")[1:])
    return runenv


def run_silently(container_id, call, extra_env=None, **kwargs):
    """Run command, an donly print output in case of failure."""
    log.debug("Silently executing call %r (with args %r)", call, kwargs)

    # Jail with container, if requested
    full_call = get_container_call(
        container_id=container_id, call=call, extra_env=extra_env, **kwargs
    )

    runenv = get_env_with_extra(extra_env)

    process = subprocess.run(
        full_call,
        **kwargs,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=set_hour_timeout,
        env=runenv,
    )
    if process.returncode != 0:
        log.error("Command %r failed with status %d", full_call, process.returncode)
        print("STDOUT: ", process.stdout.decode("utf_8"))
        print("STDERR: ", process.stderr.decode("utf_8"))
        raise Exception("Failed execution of %r %r", full_call, kwargs)


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


def get_md5_checksum(filename):
    """Return md5sum of a file."""
    with open(filename, "rb") as f:
        bytes = f.read()  # read file as bytes
        return hashlib.md5(bytes).hexdigest()


def git_descibe(dir):
    """Get git describe output for a directory."""

    git_describe_call = ["git", "describe", "--tags"]
    log.debug("Get solver version with: %r (in dir: %r)", git_describe_call, dir)
    process = subprocess.run(
        git_describe_call, cwd=dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return process.stdout.strip().decode("utf-8")


def measure_call(
    call, output_file, container_id=None, expected_code=None, extra_env=None
):
    """Run the given command, return measured performance data."""
    # Jail with container, if requested
    full_call = get_container_call(
        container_id=container_id, call=call, extra_env=extra_env
    )

    runenv = get_env_with_extra(extra_env)

    pre_stats = os.times()
    # Note: using a docker jail here reduces the precision of the measurement
    process = subprocess.run(
        full_call,
        stdout=output_file,
        stderr=subprocess.DEVNULL,
        preexec_fn=set_hour_timeout,
        env=runenv,
    )
    post_stats = os.times()
    if expected_code is not None and process.returncode != expected_code:
        log.error("Detected unexpected solver behavior when running: %r", full_call)
    return (
        {
            "cpu_time_s": (post_stats[2] + post_stats[3])
            - (pre_stats[2] + pre_stats[3]),
            "wall_time_s": post_stats[4] - pre_stats[4],
            "status_code": process.returncode,
        },
        process.stdout,
    )


class CNFgenerator(object):

    NAME = "modgen"
    REPO = "https://github.com/conp-solutions/modularityGen.git"
    VERSION = "modularityGen_v2.1"

    def __init__(self, container_id=None, cxx="g++", tool_location=None):
        self.log = logging.getLogger(self.__class__.__name__)
        self.container_id = container_id
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
        self.log.debug("Cloning '%s' from '%s'", self.NAME, self.REPO)
        commit = "cb88f2f3c7790bb478c4ff9bc9a17d7a3625591a"
        with pushd(self.workdir.name):
            clone_call = ["git", "clone", self.REPO, "."]
            self.log.debug("Cloning generator with: %r", clone_call)
            run_silently(container_id=self.container_id, call=clone_call)

            checkout_call = ["git", "reset", "--hard", commit]
            self.log.debug("Select generator commit %r", checkout_call)
            run_silently(container_id=self.container_id, call=checkout_call)

    def _build_generator(self, cxx="g++"):
        build_call = [cxx, "-O2", self.sourcefile, "-o", self.generator]  # "-Wall"
        self.log.debug("Building solver with %r", build_call)
        run_silently(container_id=self.container_id, call=build_call)

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
    SOLVER_PARAMETER = ["-no-diversify", "-no-lib-math"]

    def __init__(
        self,
        compiler=None,
        compile_flags=None,
        commit=None,
        container_id=None,
        mode=None,
        solver_location=None,
        use_solver_docker=False,
    ):
        self.log = logging.getLogger(self.__class__.__name__)
        self.build_command = None
        self.container_id = container_id
        self.mode = "debug" if mode == "debug" else "release"
        self.solver = None
        self.version = None
        self.workdir = tempfile.TemporaryDirectory(
            prefix="specsat_solver", dir=os.getcwd()
        )
        self.use_solver_docker = use_solver_docker

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
        self.solver_md5_sum = get_md5_checksum(self.solver)
        log.info("Use solver with hash sum '%s'", self.solver_md5_sum)

    def _get_solver(self, directory, commit=None):
        self.log.debug("get solver: %r", locals())
        clone_call = ["git", "clone", self.REPO, directory]
        self.log.debug("Cloning solver with: %r", clone_call)
        run_silently(container_id=self.container_id, call=clone_call)
        if commit:
            checkout_call = ["git", "reset", "--hard", commit]
            self.log.debug("Select SAT commit %r", checkout_call)
            with pushd(directory):
                run_silently(container_id=self.container_id, call=checkout_call)

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
        build_container_id = self.container_id
        if self.use_solver_docker:
            self.log.debug("Build solver with its own Dockerfile...")
            build_container_id = build_docker_container(dockerfile_dir=self.solverdir)
        run_silently(
            container_id=build_container_id, call=self.build_command, cwd=self.solverdir
        )
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

    def solve_call(self, formula_path, cores, expected_conflicts, max_conflicts=None):
        assert self.solver != None
        call = [self.solver] + self.SOLVER_PARAMETER + [f"-cores={cores}"]
        if cores > 1:
            call += ["-no-pre"]
        if max_conflicts is not None:
            call += ["-con-lim={0}".format(max_conflicts + 1)]
        elif expected_conflicts is not None:
            call += ["-con-lim={0}".format(expected_conflicts + 20000)]
        call += [formula_path]
        self.log.debug("Generated solver call: '%r'", call)
        return call

    def get_build_command(self):
        return self.build_command

    def get_md5_hash_sum(self):
        return self.solver_md5_sum

    def get_name(self):
        return self.NAME

    def get_solver_path(self):
        return self.solver

    def get_version(self):
        return self._get_version()

    def get_conflicts_from_log(self, log_file_name):
        """Get number of conflicts from solver output."""
        match_line = "c SUM stats conflicts:           :"
        with open(log_file_name) as log_file:
            all_lines = log_file.readlines()
            for line in all_lines:
                if line.startswith(match_line):
                    # extract conflicts
                    log.debug("Extracting conflicts from line '%s'", line)
                    conflicts = int(line.split(":")[2])
                    return conflicts
        # Did not find conflicts
        return None

    def validate_conflicts(self, expected_conflicts, cores, conflicts):
        """Check whether for given cores, conflicts have been detected."""
        # parallel solvers report SUM of all conflicts, hence, multiply
        match_conflicts = int(expected_conflicts) * cores

        if conflicts is not None and match_conflicts == conflicts:
            return True
        else:
            self.log.warning(
                "Expected conflicts %d for cores %d do not match detected conflicts %d - please report mismatch to author of SpecSAT and MergeSat",
                match_conflicts,
                cores,
                conflicts,
            )
            return False

    def zip_solver(self, zipname):
        """Create a zipfile and store the SAT solver binary."""
        log.info("Zip solver into '%s'", zipname)
        zipname = os.path.realpath(zipname)
        solver_path = self.get_solver_path()
        solver_name = os.path.basename(solver_path)
        pwd = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(solver_path)))
        outfile = tarfile.open(zipname, "w:xz")
        log.info("Add solver '%s' to dump", solver_name)
        outfile.add(solver_name)
        outfile.close()
        os.chdir(pwd)


class Benchmarker(object):
    BASE_WORK_DIR = FAST_WORK_DIR_NAME
    DOCKER_RUN_PREFIX = []

    def __init__(
        self,
        solver,
        generator,
        used_user_tools,
        container_id=None,
        dump_dir=None,
        pre_generated_formula_directory=None,
        measure_extra_env=None,
    ):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.debug("Get SAT Solver")
        self.solver = solver
        self.generator = generator
        self.relevant_cores = None
        self.fail_early = False
        self.used_user_tools = used_user_tools
        self.container_id = container_id
        self.dump_dir = dump_dir
        if self.dump_dir:
            self.dump_dir = os.path.realpath(self.dump_dir)
            if not os.path.exists(self.dump_dir):
                log.info("Creating output dump dir '%s'", self.dump_dir)
                os.makedirs(self.dump_dir)
        self.pre_generated_formula_directory = pre_generated_formula_directory
        self.measure_extra_env = measure_extra_env

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
            "md5_sum": self.solver.get_md5_hash_sum(),
            "version": self.solver.get_version(),
            "build_command": self.solver.get_build_command(),
        }
        report["hostinfo"] = get_host_info()
        report["relevant_cores"] = self._detect_cores()
        report["non_default_tools"] = self.used_user_tools
        report["specsat_version"] = VERSION
        return report

    def _get_benchmarks(self, only_one=False):
        # benchmark.get("expected_sequential_conflicts" if cores == 1 else "expected_1parallel_conflicts")
        benchmarks = [
            {
                "parameter": ["-s", "4900", "-n", "1000", "-m", "3000"],
                "base_sequential_cpu_time": 25,
                "expected_sequential_conflicts": 48,
                "expected_status": 10,
                "expected_benchmark_md5_hash": "0dc0f0402648bc7f2f6a5ec6973e6d7b",
                "max_parallel_conflicts": 130,
            },
            {
                "parameter": ["-s", "2400", "-n", "15000", "-m", "72500"],
                "base_sequential_cpu_time": 20,
                "expected_sequential_conflicts": 905998,
                "expected_status": 20,
                "expected_benchmark_md5_hash": "67159456029751365efc2451c9c852f2",
                "max_parallel_conflicts": 1273308,
            },
            {
                "parameter": ["-s", "4900", "-n", "1000000", "-m", "3000000"],
                "base_sequential_cpu_time": 25,
                "expected_sequential_conflicts": 352,
                "expected_status": 10,
                "expected_benchmark_md5_hash": "7a31a059e2f4986163daae16b56b6bf9",
                "max_parallel_conflicts": 956,
            },
            {
                "parameter": ["-s", "3900", "-n", "10000", "-m", "38000"],
                "base_sequential_cpu_time": 35,
                "expected_sequential_conflicts": 602372,
                "expected_status": 10,
                "expected_benchmark_md5_hash": "20a7965a2b73dcfa4a40d14c34514a7f",
                "max_parallel_conflicts": 1206209,
            },
            {
                "parameter": ["-n", "2200", "-m", "9086", "-c", "40", "-s", "158"],
                "base_sequential_cpu_time": 100,
                "expected_sequential_conflicts": 452877,
                "expected_status": 10,
                "expected_benchmark_md5_hash": "4113df86bee4aef4576790b567d3be48",
                "max_parallel_conflicts": 463282,
            },
            {
                "parameter": ["-n", "45000", "-m", "171000", "-c", "40", "-s", "100"],
                "base_sequential_cpu_time": 100,
                "expected_sequential_conflicts": 2110052,
                "expected_status": 10,
                "expected_benchmark_md5_hash": "b07fdf45f216f777b88146f7a30d0f43",
                "restriction": "sequential",
            },
            {
                "parameter": ["-n", "52500", "-m", "194250", "-c", "40", "-s", "100"],
                "base_sequential_cpu_time": 100,
                "expected_status": 10,
                "expected_benchmark_md5_hash": "ea8a0cd41f8d79367047070a5fbc9c3b",
                "max_parallel_conflicts": 2547169,
                "restriction": "parallel",
            },
        ]
        return benchmarks if not only_one else [benchmarks[0]]

    def _get_formula_basename_from_benchmark(self, benchmark):
        return "modgen_gen_{}.cnf".format("_".join(benchmark["parameter"]))

    def get_formula_for_benchmark(self, benchmark):
        """Return path to formula file that matches the file we are looking for."""

        if not self.pre_generated_formula_directory:
            raise ValueError(
                "Cannot extract pre-generated formulas if no directory is given"
            )

        basename = self._get_formula_basename_from_benchmark(benchmark)

        log.debug(
            "Look for formula file %s in directory %s",
            basename,
            self.pre_generated_formula_directory,
        )
        full_path = os.path.join(self.pre_generated_formula_directory, basename)

        if not os.path.exists(full_path):
            raise ValueError(
                f"Pregenerated formula {basename} does not exist in given directory"
            )

        return full_path

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

            # in case we generate the formulas, re-use the same file
            formula_path = os.path.join(self.BASE_WORK_DIR, "input.cnf")
            output_path = os.path.join(self.BASE_WORK_DIR, "output.log")

            detected_failure = False
            for benchmark in benchmarks:
                if detected_failure and self.fail_early:
                    self.log.warning("Stopping execution due to detected error")
                    break

                log.info("Solving benchmark %r", benchmark)
                if not self.pre_generated_formula_directory:
                    self.generator.generate(formula_path, benchmark["parameter"])
                    if self.dump_dir:
                        basename = self._get_formula_basename_from_benchmark(benchmark)
                        full_output_path = os.path.join(self.dump_dir, basename)
                        if not os.path.exists(full_output_path):
                            log.debug("Store input file as %s", full_output_path)
                            shutil.copyfile(formula_path, full_output_path)
                else:
                    formula_path = self.get_formula_for_benchmark(benchmark)
                benchmark_md5_hash = get_md5_checksum(formula_path)
                log.debug(
                    "Used benchmark has hash sum: %s, based on parameters %r",
                    benchmark_md5_hash,
                    benchmark["parameter"],
                )
                expected_benchmark_md5_hash = benchmark["expected_benchmark_md5_hash"]
                if benchmark_md5_hash != expected_benchmark_md5_hash:
                    log.error(
                        "Did not generate benchmark with known hash sum (benchmark: '%s', expected: '%s')",
                        benchmark_md5_hash,
                        expected_benchmark_md5_hash,
                    )
                    detected_failure = True
                    # TODO: create list of detected errors for error summary

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
                    solve_call = self.solver.solve_call(
                        formula_path,
                        cores,
                        benchmark.get("max_parallel_conflicts")
                        if cores != 1
                        else benchmark.get("expected_sequential_conflicts"),
                    )
                    log.debug(
                        "Solving formula %r and cores %d with solving call %r",
                        benchmark,
                        cores,
                        solve_call,
                    )
                    log.debug("Extra env for solving: %r", self.measure_extra_env)
                    with open(output_path, "w") as output_file:
                        solve_result, _ = measure_call(
                            solve_call,
                            output_file,
                            container_id=self.container_id,
                            expected_code=benchmark["expected_status"],
                            extra_env=self.measure_extra_env,
                        )
                    solve_result["validated"] = None
                    solve_result["conflicts"] = self.solver.get_conflicts_from_log(
                        output_path
                    )
                    if cores == 1:
                        if not self.solver.validate_conflicts(
                            benchmark.get("expected_sequential_conflicts"),
                            cores,
                            solve_result["conflicts"],
                        ):
                            detected_failure = True
                            solve_result["validated"] = False
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
                    if detected_failure:
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
                    if self.dump_dir is not None:
                        benchname = "".join(benchmark["parameter"])
                        dst = os.path.join(
                            self.dump_dir,
                            "output_{}_cores{}_bench{}.log".format(
                                iteration, cores, benchname
                            ),
                        )
                        log.debug("Storing solver output in file '%s'", dst)
                        shutil.copy2(output_path, dst)
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

        log.debug("Parallel stats: %r", parallel_stats)
        physical_cores = psutil.cpu_count(logical=False)
        half_cores = psutil.cpu_count(logical=False) // 2
        half_core_efficiency_factor = (
            (
                parallel_stats[half_cores]["sum_max_parallel_efficiency"]
                / parallel_stats[physical_cores]["sum_max_parallel_efficiency"]
                if parallel_stats[physical_cores]["sum_max_parallel_efficiency"]
                else 1
            )
            if half_cores in parallel_stats and half_cores != 1
            else 1
        )
        log.debug(
            "Use half cores %d and physical cores %d for half_core_efficiency_factor %f",
            half_cores,
            physical_cores,
            half_core_efficiency_factor,
        )

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
        efficiency_score = parallel_score * (
            2 - (sum_max_parallel_efficiency / num_max_parallel_runs)
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
            "half_core_efficiency_factor": half_core_efficiency_factor,
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
        print(
            "Half core factor:    {} (less is better)".format(
                half_core_efficiency_factor
            )
        )
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
        return specsat_report, detected_failure


def parse_args():
    parser = argparse.ArgumentParser(description="Run SpecSAT")
    parser.add_argument(
        "-d", "--debug", default=False, action="store_true", help="Log debug output"
    )
    parser.add_argument(
        "-v",
        "--version",
        default=False,
        action="store_true",
        help="Print version of the tool",
    )

    sub_parsers = parser.add_subparsers(help="sub-command help")
    parser_add_specsat_args(sub_parsers)
    parser_add_assess_args(sub_parsers)

    args = parser.parse_args()
    return vars(args)


def parser_add_assess_args(sub_parsers):
    parser = sub_parsers.add_parser(
        "assess", help="Assess solvers in current environment wrt being stable"
    )
    parser.set_defaults(func=run_assess_environment)

    parser.add_argument(
        "-m",
        "--max-benchmarks",
        default=-1,
        type=int,
        help="How many benchmarks to use",
    )
    parser.add_argument(
        "-o", "--output", default=None, help="Write output report to this file."
    )
    parser.add_argument(
        "-u",
        "--no-docker",
        default=False,
        action="store_true",
        help="Disable jailing with docker (more accuracy, native execution)",
    )


def parser_add_specsat_args(sub_parsers):

    parser = sub_parsers.add_parser("specsat", help="Benchmark current environment")
    parser.set_defaults(func=run_specsat)

    parser.add_argument(
        "-A",
        "--auto-archive-report",
        default=None,
        type=str,
        help="Provide hint to use when storing the report of this run as part of the archive",
    )
    parser.add_argument(
        "-D",
        "--dump-dir",
        default=None,
        type=str,
        help="Write all solver output to the given directory",
    )

    parser.add_argument(
        "-I",
        "--pre-gerated-formula-directory",
        default=None,
        type=str,
        help="Parse CNFs from given directory, instead of generating them",
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
        "-u",
        "--no-docker",
        default=False,
        action="store_true",
        help="Disable jailing with docker (more accuracy, native execution)",
    )
    parser.add_argument(
        "-R",
        "--docker-runtime",
        default=False,
        action="store_true",
        help="Jail measurement with docker (independent of build settings)",
    )
    parser.add_argument(
        "-H",
        "--docker-host-network",
        default=False,
        action="store_true",
        help="Use host network for docker commands",
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
        "--verbosity", default=0, type=int, help="Set the verbosity level"
    )
    parser.add_argument(
        "--work-dir",
        default=FAST_WORK_DIR_NAME,
        type=str,
        help="Build and run tools in this directory",
    )
    parser.add_argument(
        "-Z",
        "--zip",
        default=None,
        type=str,
        help="Zip full output into the given tar.xz file",
    )
    parser.add_argument(
        "-z",
        "--zip-solver",
        default=None,
        type=str,
        help="Zip solver binary into the given tar.xz file",
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
        "--sat-commit", default="v3.3.2", help="Use this commit of the SAT solver"
    )
    parser.add_argument("--sat-compiler", default=None, help="Use this compiler as CXX")
    parser.add_argument(
        "--sat-compile-flags",
        default=None,
        help="Add this string to CXXFLAGS and LDFLAGS",
    )
    parser.add_argument(
        "--sat-measure-extra-env",
        default=None,
        action="append",
        help="Add this value to the environment, also inside docker (of the form key=value)",
    )
    parser.add_argument(
        "--sat-mode",
        default="release",
        choices=["release", "debug"],
        help="Use solver in release or debug mode",
    )
    parser.add_argument(
        "--sat-use-solver-docker",
        default=False,
        action="store_true",
        help="Compile SAT solver using it's local Dockerfile",
    )
    parser.add_argument(
        "-S",
        "--solver-location",
        default=None,
        type=str,
        help="Location of SAT solver to be used",
    )


def write_report(report, args, tarxzdump):
    """Write output files based on args."""
    nick_name = args.get("nick_name")
    output_report = copy.deepcopy(report)
    output_report["SpecSAT"]["cli_args"] = args
    # Add nickname field to report
    if nick_name:
        output_report["nick_name"] = nick_name

    # Write generic report
    report_file = args.get("report")
    if report_file:
        with open(report_file, "w") as f:
            json.dump(output_report, f, indent=4, sort_keys=True)

    # Write full report to zip_dir
    if tarxzdump.dir():
        with open(os.path.join(tarxzdump.dir(), "full_report.json"), "w") as f:
            json.dump(output_report, f, indent=4, sort_keys=True)

    # Write output file
    output_file = args.get("output")
    output_report["SpecSAT"].pop("raw_runs")
    if output_file:
        with open(output_file, "w") as f:
            json.dump(report, f, indent=4, sort_keys=True)

    # Automatically create archivable report based on host info
    auto_archive_hint = args.get("auto_archive_report")
    if auto_archive_hint is not None:
        log.debug("Auto archiving report with hint '%s'", auto_archive_hint)
        hostinfo = get_host_info()
        # Generate file directory and name based on host details
        archive_dir_list = [
            x.replace(" ", "_")
            for x in [
                "archive",
                VERSION,
                hostinfo["cpu_info"]["brand_raw"],
                hostinfo["platform"],
            ]
        ]
        archive_dir = os.path.join(*archive_dir_list)
        lite = "lite" if output_report["SpecSAT"].get("lite_variant", False) else "full"
        valid_tools = (
            "customized"
            if output_report["SpecSAT"]["non_default_tools"]
            else "standard"
        )
        archive_base_list = (
            [lite, valid_tools, auto_archive_hint]
            if auto_archive_hint
            else [lite, valid_tools]
        )
        archive_base = "_".join(archive_base_list)
        archive_base_name = "{}.json".format(archive_base)
        log.debug("Attempting to store archive with base name '%s'", archive_base_name)
        target_name = os.path.join(archive_dir, archive_base_name)
        # Create output directory, if it does not exist yet
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)
        # Check whether file already exists, else use temporary file name with prefix)
        if os.path.exists(target_name):
            new_target_name = tempfile.NamedTemporaryFile(
                dir=archive_dir, delete=False, prefix=archive_base, suffix=".json"
            ).name
            log.warning(
                "Target archive path '%s' already exists, using a temporary file name: '%s'",
                target_name,
                new_target_name,
            )
            target_name = new_target_name
        # Write report to the archive file
        log.info("Writing archive to '%s'", target_name)
        with open(target_name, "w") as f:
            json.dump(report, f, indent=4, sort_keys=True)


def build_tools(args, container_id=None):
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
            container_id=container_id,
            cxx=args.get("generator_cxx"),
            tool_location=generator_location,
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
            container_id=container_id,
            mode=args.get("sat_mode"),
            solver_location=solver_location,
            use_solver_docker=args.get("sat_use_solver_docker"),
        )
    return (
        satsolver,
        generator,
        generator_location is not None or solver_location is not None,
    )


class TarXZDump(object):
    """Support compressing a directory, which can be filled during the life time of this object."""

    def __init__(self, zip_output=None):
        self.zip_output = zip_output
        self.fileHandler = None
        self.zip_tmp_dir = None
        self.zip_dir = None
        if not zip_output:
            return None
        self._init_data()

    def _init_data(self):
        """Setup directory to store files to be zipped later. Also, add log handler."""
        log.debug("Detected ZIP file to write to: '%s'", self.zip_output)
        # add log handler and store in file
        self.zip_tmp_dir = tempfile.TemporaryDirectory(prefix="SpecSat_zip")
        self.zip_dir = os.path.join(self.zip_tmp_dir.name, ZIP_BSAE_DIRNAME)
        os.mkdir(self.zip_dir)
        log.debug("Write output for zipping into directory '%s'", self.zip_dir)
        # Also collect logging

        logFormatter = logging.Formatter(
            "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
        )
        self.fileHandler = logging.FileHandler(os.path.join(self.zip_dir, "output.log"))
        self.fileHandler.setFormatter(logFormatter)
        self.fileHandler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(self.fileHandler)

    def _tear_down(self):
        """Make object unusable."""
        self.zip_dir = None

    def dir(self):
        """Dreturn name of the directory we will zip later."""
        return self.zip_dir

    def finalie_tarxz(self):
        """Compress the data of the directory."""

        if not self.zip_dir:
            return

        name = os.path.basename(self.zip_dir)
        zipname = (
            self.zip_output
            if self.zip_output.endswith("tar.xz")
            else f"{self.zip_output}tar.xz"
        )
        log.info("Creating ZIP file with dump: '%s'", zipname)
        zipname = os.path.realpath(zipname)

        # Make sure the files we zip have all content
        self.fileHandler.flush()

        # Finally, create the archive file
        pwd = os.getcwd()
        os.chdir(os.path.join(self.zip_dir, os.pardir))
        log.debug("Change to directory '%s' for compression", os.getcwd())
        outfile = tarfile.open(zipname, "w:xz")
        log.debug("Add directory '%s' to dump", name)
        outfile.add(name)
        outfile.close()
        os.chdir(pwd)
        log.debug("Change back to directory '%s' after compression", os.getcwd())
        self._tear_down()


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

    log.debug("args: %r", args)

    function = args["func"]
    ret = function(args)
    log.info("Finished command with %d", ret)


def download_hashed_file(file_hash):
    """Retrieve competition file from hash."""
    download_command = [
        "wget",
        "--content-disposition",
        f"https://gbd.iti.kit.edu/file/{file_hash}",
    ]
    run_silently(None, download_command)


class SAT_solver_git(object):

    CLONE_CMD = ["git", "clone"]
    CHECKOUT_CMD = ["git", "checkout"]

    def __init__(self):
        pass

    def initialize_solver(self, docker_container_id: str = None, dir: str = None):
        self._temp_dir = None if dir else tempfile.TemporaryDirectory()
        self._dir = dir if dir else self._temp_dir.name
        self.container_id = docker_container_id
        self.binary_path = self.download_and_compile()
        self.git_describe = git_descibe(os.path.join(self._dir, self.SOLVER_NAME))

    def download_and_compile(self):
        """Download and build, only works if other fields are given in inherited objects."""
        log.debug("Download and build %s in directory %s", self.SOLVER_NAME, self._dir)
        with pushd(self._dir):
            # Clone
            cmd = self.CLONE_CMD[0:]
            cmd += [self.REPO, self.SOLVER_NAME]
            run_silently(self.container_id, call=cmd)

            # In repository,
            with pushd(self.SOLVER_NAME):
                # Checkout
                checkout_cmd = self.CHECKOUT_CMD[0:]
                checkout_cmd.append(self.COMMIT)
                run_silently(self.container_id, call=checkout_cmd)

                for command in self.BUILD_COMMANDS:
                    run_silently(self.container_id, call=command)

                # Return path to actual binary
                return os.path.realpath(self.BINARY_PATH)


class cadical_watchsat_lto_solver(SAT_solver_git):

    REPO = "https://github.com/conp-solutions/cadical"
    COMMIT = "301cc5a8290801ff64867ba4673287411ef2432d"
    BUILD_COMMANDS = [["./configure", "-static"], ["make", "-j", "8"]]
    SOLVER_NAME = "cadical-watchsat-flto"
    BINARY_PATH = "build/cadical"

    def __init__(self, docker_container_id: str = None, dir: str = None):
        self.initialize_solver(docker_container_id=docker_container_id, dir=dir)

    def solve_call(self, cnf_file):
        return [self.binary_path, cnf_file]

    def get_conflicts(self, stdout):
        """Extract conflicts from output, looking like:
c --- [ statistics ] ---------------------------------------------------------
c 
c chronological:              2658        36.29 %  of conflicts
c conflicts:                  7325     33956.21    per second
        """
        conflicts = -1
        for line in stdout.splitlines():
            if line.startswith("c conflicts:"):
                conflicts = int(line.split()[2])
        # return the last hit for this pattern
        return conflicts


class kissat_bulky_solver(SAT_solver_git):

    REPO = "https://github.com/arminbiere/kissat.git"
    COMMIT = "c5cce1b9b35bc74566f192c43620bff03e4a9ff6"
    BUILD_COMMANDS = [["./configure", "-static"], ["make", "-j", "8"]]
    SOLVER_NAME = "kissat"
    BINARY_PATH = "build/kissat"

    def __init__(self, docker_container_id: str = None, dir: str = None):
        self.initialize_solver(docker_container_id=docker_container_id, dir=dir)

    def solve_call(self, cnf_file):
        return [self.binary_path, cnf_file]

    def get_conflicts(self, stdout):
        """Extract conflicts from output, looking like:
c ---- [ statistics ] --------------------------------------------------------
c
c backbone_implied:                        94                3.36 per unit
c backbone_units:                          28                2 %  variables
c clauses_learned:                      10895               87 %  conflicts
c conflicts:                            12562            52479.43 per second
c decisions:                            24405                1.94 per conflict
        """
        conflicts = -1
        for line in stdout.splitlines():
            if line.startswith("c conflicts:"):
                conflicts = int(line.split()[2])
        # return the last hit for this pattern
        return conflicts


class mergesat(SAT_solver_git):

    REPO = "https://github.com/conp-solutions/mergesat.git"
    COMMIT = "258afd6ae3320a2cdea893a23c9cb8b2f093bafe"
    BUILD_COMMANDS = [["make", "r", "-j", "8"]]
    SOLVER_NAME = "mergesat"
    BINARY_PATH = "build/release/bin/mergesat"

    def __init__(self, docker_container_id: str = None, dir: str = None):
        self.initialize_solver(docker_container_id=docker_container_id, dir=dir)

    def solve_call(self, cnf_file):
        return [self.binary_path, "-no-lib-math", cnf_file]

    def get_conflicts(self, stdout):
        """Extract conflicts from output, looking like:
c ===============================================================================
c simplification        : 4 elim.vars,  2 subsumed
c strengthening         : 3 lits, 0 all-candidates, 0 all-used
c restarts              : 12
c conflicts             : 5473           (23513 /sec)
c decisions             : 16558          (0.00 % random) (71137 /sec)
c propagations          : 590045         (2534971 /sec)
c conflict literals     : 96779          (22.89 % deleted)
        """
        conflicts = -1
        for line in stdout.splitlines():
            if line.startswith("c conflicts             : "):
                conflicts = int(line.split()[3])
        # return the last hit for this pattern
        return conflicts


def run_assess_environment(args):

    log.debug("Run assessing environment with args %r", args)

    max_benchmarks = args.get("max_benchmarks", -1)
    output_file = args.get("output")
    return_code = 0

    if max_benchmarks != -1:
        log.info("Run on the first %d benchmarks", max_benchmarks)

    """
    Check CaDiCaL-watchsat-lto, Kissat-bulky and MergeSat on the first formulas that took 3600 seconds
    in the competition together (data from main track result columns)

    hash	benchmark	verified-result	CaDiCaL-watchsat-lto	MergeSat 4.0-rc-rc3	kissat-sc2022-bulky	SUM	TOTAL_SUM
    d87714e099c66f0034fb95727fa47ccc	Wallace_Bits_Fast_2.cnf.cnf.xz	sat	1.14809	2.81604	2.20632	6.17045	11.19281
    0616ca6b1e0ae639d12e19ed310963a9	satcoin-genesis-SAT-4.cnf.xz	sat	1.3179	22.7107	19.8417	43.8703	55.06311
    6d815089fb4095208754622b4c44a8f7	Carry_Bits_Fast_12.cnf.cnf.xz	sat	3.76298	40.6661	3.02327	47.45235	102.51546
    00a62eb89afbf27e1addf8f5c437da9b	ortholatin-7.cnf.xz	sat	3.43583	34.1585	16.1187	53.71303	156.22849
    49f5d904bb2e762219287d33f5621971	ITC2021_Late_1.xml.cnf.xz	sat	15.2517	58.8198	25.6195	99.691	255.91949
    5099718e4b5b8fe2ba7f8b919fdfde2d	satcoin-genesis-SAT-5.cnf.xz	sat	1.2681	23.5836	85.5058	110.3575	366.27699
    0dc468413b65339fad7e4f981bf2fb1e	reconf20_61_myciel4_2.cnf.xz	unsat	32.1735	58.0136	23.1146	113.3017	479.57869
    8bb5819a23a8f1dff851d608bec008ca	ITC2021_Late_6.xml.cnf.xz	sat	39.3672	79.8123	15.9397	135.1192	614.69789
    807133f4461a11e39a390dfcf67a4fc6	summle_X11113_steps8_I1-2-2-4-4-8-25-100.cnf.xz	sat	57.1315	39.1326	66.1284	162.3925	777.09039
    bb80971144a5423532aea9424dade891	div-mitern174.cnf.xz	unsat	37.1634	105.256	38.895	181.3144	958.40479
    436d00b6d41912e800e9855d08c75139	manol-pipe-f7nidw.cnf.xz	unsat	38.0336	86.7552	60.7459	185.5347	1143.93949
    920bbe559da4ea37fd636605316a791c	SCPC-500-19.cnf.xz	unsat	34.0026	137.37	36.1053	207.4779	1351.41739
    3176c9024f756e99dbee852e5c0aa578	Break_triple_12_30.xml.cnf.xz	sat	113.311	92.2417	11.9348	217.4875	1568.90489
    d3c07914f3ebb42906b986aa431243af	summle_X8646_steps8_I1-2-2-4-4-8-25-100.cnf.xz	sat	131.17	30.796	59.9684	221.9344	1790.83929
    608941b989227d93c62b3a9b4280011b	summle_X8639_steps8_I1-2-2-4-4-8-25-100.cnf.xz	sat	146.001	40.8328	49.9949	236.8287	2027.66799
    e28cf003c086c99f3a3d9a90aebf8ed1	intel046_Iter124.cnf.xz	sat	79.094	125.537	33.7758	238.4068	2266.07479
    fa88b447f6b41b04686085480678affe	UNSAT_H_instances_childsnack_p12.hddl_1.cnf.xz	unsat	94.0074	81.4267	64.8844	240.3185	2506.39329
    c0df94532b1ca8a705d3af05378f377d	SCPC-500-20.cnf.xz	unsat	44.3795	151.448	44.913	240.7405	2747.13379
    5d357f1e4b5abc96c90d305f629edd5c	ps_200_323_70.cnf.xz	unsat	111.817	102.158	34.5328	248.5078	2995.64159
    3b97191d94abf3a5e97d6359d886d9df	summle_X111121_steps7_I1-2-2-4-4-8-25-100.cnf.xz	sat	140.091	66.0687	42.8702	249.0299	3244.67149
    1ebab99edff4c0470c8aa753bb9d3353	summle_X8637_steps8_I1-2-2-4-4-8-25-100.cnf.xz	sat	120.212	107.106	26.6837	254.0017	3498.67319
    """

    assess_report = dict()
    assess_report["hostinfo"] = get_host_info()

    benchmarks = [
        "d87714e099c66f0034fb95727fa47ccc",
        "0616ca6b1e0ae639d12e19ed310963a9",
        "6d815089fb4095208754622b4c44a8f7",
        "00a62eb89afbf27e1addf8f5c437da9b",
        "49f5d904bb2e762219287d33f5621971",
        "5099718e4b5b8fe2ba7f8b919fdfde2d",
        "0dc468413b65339fad7e4f981bf2fb1e",
        "8bb5819a23a8f1dff851d608bec008ca",
        "807133f4461a11e39a390dfcf67a4fc6",
        "bb80971144a5423532aea9424dade891",
        "436d00b6d41912e800e9855d08c75139",
        "920bbe559da4ea37fd636605316a791c",
        "3176c9024f756e99dbee852e5c0aa578",
        "d3c07914f3ebb42906b986aa431243af",
        "608941b989227d93c62b3a9b4280011b",
        "e28cf003c086c99f3a3d9a90aebf8ed1",
        "fa88b447f6b41b04686085480678affe",
        "c0df94532b1ca8a705d3af05378f377d",
        "5d357f1e4b5abc96c90d305f629edd5c",
        "3b97191d94abf3a5e97d6359d886d9df",
        "1ebab99edff4c0470c8aa753bb9d3353",
    ]

    # TODO: go to working directory

    assess_container_id = None
    if not args.get("no_docker", True):
        assess_artifacts_dir = "assess_artifacts"
        specsat_dir = os.path.dirname(os.path.abspath(__file__))
        dockerfile_dir = os.path.join(specsat_dir, assess_artifacts_dir)
        log.info("Building assess docker container from directory %s", dockerfile_dir)
        assess_container_id = build_docker_container(dockerfile_dir=dockerfile_dir)

    solvers = dict()
    solvers["cadical-watchsat-flto"] = cadical_watchsat_lto_solver(
        docker_container_id=assess_container_id
    )
    solvers["kissat_bulky"] = kissat_bulky_solver(
        docker_container_id=assess_container_id
    )
    solvers["mergesat"] = mergesat(docker_container_id=assess_container_id)
    log.info("Work with solvers: %r", solvers)

    dir = tempfile.TemporaryDirectory()

    assess_report["solvers"] = {}
    for solver_name, solver in solvers.items():
        md5_sum = get_md5_checksum(solver.binary_path)
        assess_report["solvers"][solver_name] = {
            "md5sum": md5_sum,
            "git_describe": solver.git_describe,
        }

    assess_report["benchmarks"] = {}
    assessed_benchmarks = 0
    try:
        for benchmark in benchmarks:

            assessed_benchmarks += 1
            if max_benchmarks != -1 and assessed_benchmarks > max_benchmarks:
                log.info("Stopping assessment, as we reached specified maximum")
                break

            with pushd(dir.name):
                download_hashed_file(benchmark)
                zipped_cnf_files = glob.glob("*.cnf.xz")
                log.debug("zipped cnf: %r", zipped_cnf_files)
                assert len(zipped_cnf_files) == 1, "only one file should be downloaded"
                zipped_cnf = os.path.realpath(zipped_cnf_files[0])

                assess_report["benchmarks"][benchmark] = {
                    "name": os.path.basename(zipped_cnf)
                }

                log.info(
                    "On benchmark %s, working with solvers: %r", benchmark, solvers
                )
                for solver_name, solver in solvers.items():
                    log.info("Run solver %s on benchmark %s", solver_name, zipped_cnf)
                    full_call = solver.solve_call(zipped_cnf)
                    log.debug("Execute solver call: %r", full_call)

                    data = {}
                    try:
                        process = subprocess.run(
                            full_call,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL,
                            preexec_fn=set_5m_timeout,
                        )

                        data, stdout = measure_call(
                            full_call,
                            subprocess.PIPE,
                            container_id=None,
                            expected_code=None,
                            extra_env=None,
                        )

                        stdout = process.stdout.decode()
                        log.debug(
                            "Call output %s with return code %d",
                            stdout,
                            data["status_code"],
                        )
                        data["conflicts"] = solver.get_conflicts(stdout)
                    except Exception as e:
                        log.debug("Failed solver execution with exception %r", e)
                        log.error(
                            "Failed to execute solver %s on benchmark %s",
                            solver_name,
                            zipped_cnf,
                        )
                        return_code = 1
                    assess_report["benchmarks"][benchmark][solver_name] = data

                # Cleanup this iteration
                os.remove(zipped_cnf)

    except Exception as e:
        log.exception("Stopped call with exception: %r", e)
        return_code = 1
    log.info("Full report: %s", json.dumps(assess_report, indent=2))

    if output_file:
        log.info("Writing assessment report to file: %s", output_file)
        with open(output_file, "w") as f:
            json.dump(assess_report, f, indent=4, sort_keys=True)

    return return_code


def run_specsat(args):
    log.info("Run run_specsat, with args %r", args)

    if args.get("docker_host_network"):
        log.debug("Using host network for docker")
        DOCKER_NETWORK = ["--network=host"]

    # prepare storing all data in a zip file
    zipdump = TarXZDump(args.get("zip"))

    log.info("SpecSAT 2021, version %s", VERSION)
    log.debug("Starting benchmarking with args: %r", args)

    container_id = None
    if not args.get("no_docker", True) or args.get("docker_runtime", False):
        container_id = build_docker_container()

    satsolver, generator, used_user_tools = build_tools(args, container_id=container_id)

    if args.get("zip_solver"):
        satsolver.zip_solver(args.get("zip_solver"))

    output_dir = args.get("dump_dir")
    output_dir = output_dir if output_dir is not None else zipdump.dir()
    # make sure we use an absolute path for the provided input
    pre_generated_formula_directory = args.get("pre_generated_formula_directory")
    pre_generated_formula_directory = (
        None
        if pre_generated_formula_directory is None
        else os.path.realpath(pre_generated_formula_directory)
    )
    # create benchmarking object
    benchmarker = Benchmarker(
        solver=satsolver,
        generator=generator,
        used_user_tools=used_user_tools,
        container_id=container_id if args.get("docker_runtime", False) else None,
        dump_dir=output_dir,
        pre_generated_formula_directory=pre_generated_formula_directory,
        measure_extra_env=args.get("sat_measure_extra_env"),
    )
    if zipdump.dir() is not None and zipdump.dir() != output_dir:
        shutil.copytree(output_dir, zipdump.dir(), dirs_exist_ok=True)
    report, failure = benchmarker.run(
        iterations=args.get("iterations"),
        lite=args.get("lite", False),
        verbosity=args.get("verbosity"),
    )
    write_report(report, args, tarxzdump=zipdump)
    zipdump.finalie_tarxz()

    log.info("Finished SpecSAT")
    return 1 if failure else 0


if __name__ == "__main__":
    sys.exit(main())
