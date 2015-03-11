#!/bin/sh

for dig in `ls /data/projects/mnist/data/train`; do
    for file in `ls /data/projects/mnist/data/train/$dig`; do
	id=`echo $file | sed 's/\.png//'`
	name=$id-$dig-00.png
	cp /data/projects/mnist/data/train/$dig/$file /data/projects/mnist/data/whole/$name
    done
done

for dig in `ls /data/projects/mnist/data/valid`; do
    for file in `ls /data/projects/mnist/data/valid/$dig`; do
	id=`echo $file | sed 's/\.png//'`
	name=$id-$dig-00.png
	cp /data/projects/mnist/data/valid/$dig/$file /data/projects/mnist/data/whole/$name
    done
done

for dig in `ls /data/projects/mnist/data/test`; do
    for file in `ls /data/projects/mnist/data/test/$dig`; do
	id=`echo $file | sed 's/\.png//'`
	name=$id-$dig-00.png
	cp /data/projects/mnist/data/test/$dig/$file /data/projects/mnist/data/whole/$name
    done
done
