To compute flops of all operations, we use [torchstat](https://github.com/Swall0w/torchstat). As there are some operations disappeared from original implementation, We modified it to add those(See modified_package).

### Total time counting

```
python totaltime_test.py 3
```

### Flops computation

```
python flops_count.py 3
```

### Time for each operations

```
python time_detail.py 3
```

