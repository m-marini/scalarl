# Maze environment

[TOC]

## Abstract

The environment consists of a plan grid of $10 \times 10 $ cells.

## Initial position

At initial step the subject is located in the initial cell.

## Goal

The goal of subject is to move to near cell to reach the target cell.

## Wall constraint

Some of the cell is constituted by wall and the subject cannot move on them.

## Actions

At each step the subject decides which action perfom between a set of available actions debending on the subject position.
At most There are eight availabe actions corresponding to the eight adjactent cells.

The enumeration of these actions is

| Direction       | Value |
|-----------------|------:|
| N - North       |     0 |
| NE - North east |     1 |
| E - East        |     2 |
| SE - South east |     3 |
| S - South       |     4 |
| SW - South west |     5 |
| W - West        |     6 |
| NW - North west |     7 |

## Rewards

When the subject reach the target cell it gets a reward of 10 units
if no  move has been done then it receives a reword of -2 unit.
else it receives a negative reword of distance movement

## Input signals

The input signals consist of 100 input units representing each cell position where the 1 value is set for the current subject position.

## Available action signals

The available action signals consist of 8 input units representing each action where 1 value is set for the available actions.

## Output signals

The output signals consists of 8 output units representing the estimation of advantage action value. The higer value of the output indicates the action to be taken.
