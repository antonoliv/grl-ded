#!/bin/sh

cpufreq --set-turbo 1
cpufreq --set-preference performance
cpufreq --set-governor performance
