#!/bin/bash

# Find all folders starting with "p1" in the current directory
for folder in /remote-home/limengxun/tujie/NeAT/Experiments/p1_*
do
    # Check if the folder exists and is a directory
    if [ -d "$folder" ]; then
        python eval.py --exp $folder --ds /remote-home/limengxun/tujie/cbctrec2/cbctrec/.tmpdir/data/final_test/train_data-p1.npz
    fi
done


for folder in /remote-home/limengxun/tujie/NeAT/Experiments/p3_*
do
    # Check if the folder exists and is a directory
    if [ -d "$folder" ]; then
        python eval.py --exp $folder --ds /remote-home/limengxun/tujie/cbctrec2/cbctrec/.tmpdir/data/final_test/P0194247-p3.npz
    fi
done

for folder in /remote-home/limengxun/tujie/NeAT/Experiments/p4_*
do
    # Check if the folder exists and is a directory
    if [ -d "$folder" ]; then
        python eval.py --exp $folder --ds /remote-home/limengxun/tujie/cbctrec2/cbctrec/.tmpdir/data/final_test/P0275935-p4.npz
    fi
done

for folder in /remote-home/limengxun/tujie/NeAT/Experiments/p5_*
do
    # Check if the folder exists and is a directory
    if [ -d "$folder" ]; then
        python eval.py --exp $folder --ds /remote-home/limengxun/tujie/cbctrec2/cbctrec/.tmpdir/data/final_test/P0280336-p5.npz
    fi
done

for folder in /remote-home/limengxun/tujie/NeAT/Experiments/p6_*
do
    # Check if the folder exists and is a directory
    if [ -d "$folder" ]; then
        python eval.py --exp $folder --ds /remote-home/limengxun/tujie/cbctrec2/cbctrec/.tmpdir/data/final_test/P0047120-p6.npz
    fi
done

for folder in /remote-home/limengxun/tujie/NeAT/Experiments/p7_*
do
    # Check if the folder exists and is a directory
    if [ -d "$folder" ]; then
        python eval.py --exp $folder --ds /remote-home/limengxun/tujie/cbctrec2/cbctrec/.tmpdir/data/final_test/P0118783-p7.npz
    fi
done

for folder in /remote-home/limengxun/tujie/NeAT/Experiments/p8_*
do
    # Check if the folder exists and is a directory
    if [ -d "$folder" ]; then
        python eval.py --exp $folder --ds /remote-home/limengxun/tujie/cbctrec2/cbctrec/.tmpdir/data/final_test/P0389692-p8.npz
    fi
done

for folder in /remote-home/limengxun/tujie/NeAT/Experiments/p9_*
do
    # Check if the folder exists and is a directory
    if [ -d "$folder" ]; then
        python eval.py --exp $folder --ds /remote-home/limengxun/tujie/cbctrec2/cbctrec/.tmpdir/data/final_test/P0405910-p9.npz
    fi
done

for folder in /remote-home/limengxun/tujie/NeAT/Experiments/p10_*
do
    # Check if the folder exists and is a directory
    if [ -d "$folder" ]; then
        python eval.py --exp $folder --ds /remote-home/limengxun/tujie/cbctrec2/cbctrec/.tmpdir/data/final_test/P0407419-p10.npz
    fi
done

for folder in /remote-home/limengxun/tujie/NeAT/Experiments/p11_*
do
    # Check if the folder exists and is a directory
    if [ -d "$folder" ]; then
        python eval.py --exp $folder --ds /remote-home/limengxun/tujie/cbctrec2/cbctrec/.tmpdir/data/final_test/P0424477-p11.npz
    fi
done