#!/bin/bash

for f in experiments/fc/*T2/*.fc; do
    echo "$f"
	tsp timeout 300s ./runOne.sh $f $@
done
