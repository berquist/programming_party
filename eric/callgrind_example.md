Run `callgrind` on the binary:

```
valgrind --tool=callgrind ./project3

```

then parse the output:

```
callgrind_annotate --auto=yes --inclusive=no callgrind.out.25606
```
