SpecSAT
===============================================================================

The goal of SpecSAT is to understand how well a SAT solver performs on a given
hardware platform. This is done be running a fixed SAT solver on a fixed set
of benchmark formulas, and measuring the wall clock time this computation takes.

As parallel SAT solvers allow to run different solving configurations in
parallel, we also measure how well a parallel solver performs on the given
architecture - both when using the number of all CPUS, as well as using only
physical CPUS (i.e. excluding logical CPUs).

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

Norbert Manthey
