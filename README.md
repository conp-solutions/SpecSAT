# SpecSAT

Norbert Manthey <nmanthey@conp-solutions.com>

## Goal

The goal of SpecSAT is to understand how well a SAT solver performs on a given
hardware platform. This is done be running a fixed SAT solver on a fixed set
of benchmark formulas, and measuring the wall clock time this computation takes.

As parallel SAT solvers allow to run different solving configurations in
parallel, we also measure how well a parallel solver performs on the given
architecture - both when using the number of all CPUS, as well as using only
physical CPUS (i.e. excluding logical CPUs).

## Quick Start

To just benchmark your hardware, use the below command. This will run all
measurements inside a python virtual environment. A report of your setup will
automatically be written into the 'archive' directory.

```
./SpecSAT.sh -A "" -i 3
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
