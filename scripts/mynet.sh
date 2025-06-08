#!/bin/bash
######
# Taken from https://github.com/emp-toolkit/emp-readme/blob/master/scripts/throttle.sh
######

# replace DEV=lo with your card (e.g., eth0)
DEV=lo 
if [ "$1" == "del" ]
then
	tc qdisc del dev $DEV root
fi

if [ "$1" == "lan" ]
then
tc qdisc del dev $DEV root
## bandwidth = rate = 1Gbps, token bucket size = burst = 100kB, token bucket buffer size = 1MB
tc qdisc add dev $DEV root handle 1: tbf rate 1gbit burst 100k limit 1m
## round trip time = 2 * latency = 2 * delay = 2ms
tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 1msec
fi

if [ "$1" == "wan" ]
then
tc qdisc del dev $DEV root
## bandwidth = rate = 160Mbps, token bucket size = burst = 100kB, token bucket buffer size = 1MB
tc qdisc add dev $DEV root handle 1: tbf rate 160mbit burst 100k limit 1m
## round trip time = 2 * latency = 2 * delay = 50ms
tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 50msec
fi