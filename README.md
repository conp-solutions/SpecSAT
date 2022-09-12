# SpecSAT

Norbert Manthey <nmanthey@conp-solutions.com>

## Goal

The goal of SpecSAT is to understand how well a SAT solver performs on a given
hardware platform. This is done be running a fixed SAT solver on a fixed set
of benchmark formulas, and measuring the wall clock time this computation takes.

Besides benchmarking hardware configuration, also other features of the platform
or even the solver implementation can be benchmarked. An example list of
features is using "huge pages" for memory management, using the prefetching unit
of the CPU, using different compilation flags, or even use different layout of
the data structures within the SAT solver (without influencing its search
steps). To use a stable build, docker is used to create the SAT solver binary.

As parallel SAT solvers allow to run different solving configurations in
parallel, we also measure how well a parallel solver performs on the given
architecture - both when using the number of all CPUS, as well as using only
physical CPUS (i.e. excluding logical CPUs).

## Quick Start

To just benchmark your hardware, use the below command. This will run all
measurements inside a python virtual environment. A report of your setup will
automatically be written into the 'archive' directory. Finally, the generated
report, as well as all solver output, will be added to an archive tar file.

```
./SpecSAT.sh -A "" -i 3 -Z archive.tar.xz
```

A more complex example, to use more capabilities of the used SAT solver and to
better test the limits of the system, is to compile the SAT solver using its own
dockerfile (so that it links against a recent libc), and then use transparent
huge pages via madvise in the SAT solver. Finally, for restricted networks, the
docker commands can use the host network. The following command achieves this,
and stores the report in the specified file:

```
    ./SpecSAT.py -d \
        --sat-measure-extra-env=GLIBC_TUNABLES=glibc.malloc.hugetlb=1 \
        --sat-use-solver-docker -u \
        -H \
        -r report.json
```

## Usage

To improve the efficiency of the used solver, a different compiler for the
solver can be specified. Furthermore, solver additional compile time parameter
can be specified.

To see the available command line parameter, run:

```
./SpecSAT.py -h
```

The measured data can be strored to report files, where also a full report with
the native results can be stored:

```
./SpecSAT.py -r full-report.json -o output.json
```

### Running with no docker available

To build the SAT solver, we use a Docker container to make sure we use the same
libraries and dependencies. However, some systems that should be evaluated do
not come with a docker setup, and the user might lack administrator privileges
to add docker. To work-around these situations, the tool allows to store the
used SAT solver in a ZIP file. Next, the binary in that file can be used in a
follow up run on the actual target machine. The steps are as follows:

#### Run lite mode to store the compiled solver

First, run a lite run, and build the SAT solver in a docker container. Also,
store the solver in an archive.

```
./SpecSAT.py -l -z mergesat.tar.xz -o precompile-report.json
```

#### Run actual measurement with pre-compiled solver

On the target machine, we can then receive the target file, extract it, and
then forward it to SpecSAT.

```
tar xJf mergesat.tar.xz
./SpecSAT.sh -r full-report.json -o output.json --solver-location mergesat
```

Note: the new tool will now complain about using an external solver, and
suggests the measurement might not be correct. You can compare the md5 sum of
the used binary against the precompiled binary to make sure the results are
actually correct:

On the build machine:

```
grep "md5" precompile-report.json
```

On the measurement machine:

```
grep "md5" output.json
```

### Usage for ARM Platforms

To build the SAT solver, we use a Docker container to make sure we use the same
libraries and dependencies. The used container file does not work on ARM right
now. Hence, please use the `--no-docker` parameter to disable using docker.

### Wrapper Script Extensions

To not bother with setting up the environment, a convenience wrapper is
provided. This wrapper takes care of setting up the dependencies in a virtual
environment (via venv). Furthermore, the wrapper supports two command line
parameters:

```
    <some_path>/SpecSAT.sh --run-from-install [args]
```

... will run SpecSAT.py with args `args` in the directory where the tool itself
is located. This allows to use the tool in e.g. a distributed environment.

```
    <some_path>/SpecSAT.sh --force-install [args]
```

... will run the wrapper, and re-install the python dependencies. This command
is only required, in case the python dependencies change.

## Dependencies

To execute SpecSAT, non-default python packages need to be installed. The list
of packages to be installed is listed in the file requirements.txt. This package
list can be installed via

```
python3 -m pip install --upgrade --user -U -r requirements.txt
```

Note: to not pollute the configuration of your python installation, you should
consider running SpecSAT in a virtual environment, e.g. venv. If you do not
want to maintain this venv separately, the script `SpecSAT.sh` will take care
of the extra steps automatically.

## Internals

The SpecSAT tool uses the SAT solver `MergeSat` as a SAT backend to perform
measurements. The input problems to be solved are generated with the tool
`modgen`. Both tools are considered state-of-the-art tools, so that using them
for measurements provides meaningful results.

The generated benchmarks are large enough to represent currently challenging
input sizes for a SAT solver. `MergeSat` can solve input by using multiple
compute resources. This parallel solving is implemented in a deterministic way.
For measurements, this configuration allows to re-run the parallel solving with
the exact same algorithmic steps.

To make sure the expected results are obtained, the output of the sequential
solving process is validated against the known number of search conflicts the
SAT solver will produce when solving a given formula. For the parallel search,
no such expectation is used, as the number of generated conflicts also depends
on the number of used cores.

### Licensing

SpecSAT is implemented under the MIT license. The used SAT solver `MergeSat`
also uses the MIT license. The benchmark generation tool `modgen` is distributed
via the GPL license. Please make sure these licenses fit your use case.

## Calculated Scores

The score for the hardware is computed based on the wall clock time it takes to
solve the predefined benchmarks. To compute the efficiency of the hardware with
respect to parallel SAT solving, the CPU time is also considered. For the
benchmark with the highest computation time, also the variance is computed.
Using multiple iterations of measurements is supported, and allows to reduce the
variance of the obtained data points.

### Score_Sequential

This score scales linear with the wall clock time of the sequential solving. The
smaller the obtained number, the better the platform is for sequential SAT
solving.

### Score_Full_Parallel

This score scales linear with the wall clock time of the parallel solving, when
using all available cores of the platform. The smaller the number, the better
the given platform is for parallel SAT solving.

Note: there is no linear relation between sequential and parallel score, as the
performed search steps are different.

### Score_Efficiency

This score takes the efficiency of the platform into account, and shows whether
there is potential room for improvement. In case the solver runs as efficient
as possible (efficiency = 1), the score_efficiency value is equal to the
Score__Full_Parallel value. The score decreases linear to the double of this
score depending on the measured efficiency of the solver when using all cores,
i.e.:

```
 efficiency_score = score_full_parallel * (2 - efficiency)
```

### Half_Core_Efficiency_Factor

SAT solvers are assumed to be bound by the memory bandwidth. When using all
cores of the platform, the available memory bandwidth per used core should be
lower compared to using only every second core.

This score hence is the ratio of the measured efficiency of the solver when
using half of the available cores vs using all cores, i.e.:

```
half_core_efficiency_factor = efficiency(n/2) / efficiency(n),
```

where `n` is the number of available cores.

## Contributing Report Data

Data about different hosts can be stored automatically as part of the repository
of the SpecSAT tool. To benchmark your system, and automatically store a report,
follow the steps below! Depending on your hardware, executing the measurement
might take up to two hours.

Note: the report is supposed to not contain any personal data. However, before
      making your data public, make sure you checked the details!

 0. clone the repository locally
 1. run the tool, and generate an auto-report, by running `./SpecSAT.sh -A "" -i 3`
 2. create a git branch in your local repository (e.g. `git checkout -b ...`)
 3. add the new report file to the branch
 4. create a pull request in the main repository with your change

When contributing data to the SpecSAT repository, make sure you provide your
change with the MIT license.

## Reporting Errors

In case the tool returns with a non-zero exit code, and you want to report an
issue, please make sure to run the tool with the default setup. If the error
still persist, one of the following error conditions might be true:

 1. The used platform is not supported (currently only x86_64)
    Feel free to extend SpecSAT to support your platform. Further extensions are
    not considered at the time of writing.
 2. The used platform triggers a different behavor in the used solver.
 3. Something else.

In case you want to report an issue, feel free to create an issue in github.
Please re-run SpecSAT in debug mode, and collect all output in an archive. Add
this archive to the github issue.

Submit an issue here: https://github.com/conp-solutions/SpecSAT/issues

Use these parameters to enable debugging and create an archive: `-d -Z specsat.tar.xz`

Full command to create relevant output for investigation:

```
./SpecSAT.sh -A "" -d -i 3 -Z archive.tar.xz
```
