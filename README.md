# Location and Positional Accuracy Determiner

# TruePin

**Author:** Giwin Vincent Edwin Omesh

## Overview

This project explores the problem of **spatial repositioning of places in large-scale mapping datasets**, with a focus on identifying, measuring, and correcting positional inaccuracies in point-based location data.

Modern global maps are frequently updated, yet **pin placement errors**—where a place is represented at an incorrect or suboptimal location—remain a persistent challenge. These inaccuracies can negatively affect navigation, search relevance, routing, and downstream spatial analytics.

This repository contains a research-driven prototype system that investigates **what a “correct” location actually means**, how to measure deviation from it, and how models can be built to automatically reposition places more accurately.

## Problem Statement

Mapping datasets often store a single coordinate per place, but real-world locations can be defined in multiple ways (e.g. building centroid, entrance, parcel center, or activity hotspot). Without a clear definition of “ground truth,” spatial errors are difficult to quantify or fix.

This project aims to:
- Identify suitable definitions for pin placement
- Quantify spatial offsets in existing place data
- Prototype modeling approaches to improve positional accuracy

## Project Goals

- **Define ground-truth pin locations** for places and evaluate their strengths and weaknesses
- **Measure spatial offset** between existing map data and ground truth
- **Prototype repositioning models** that select more accurate coordinates
- **Provide recommendations** for scalable future improvements

## Data

- Uses a provided sample of ~5,000 places from the Overture dataset
- Constructs a manually curated **ground-truth dataset** (target size: 500–1,000 places)
- Compares current place coordinates against validated reference locations

## Research Questions

- What are the most meaningful definitions of a place’s location?
- How should positional error be measured in a consistent and interpretable way?
- What modeling approaches work best for spatial repositioning?
- How can geographic space be discretized effectively for prediction?
- Which spatial and contextual features are most informative?

## Deliverables

- A working definition of accurate pin placement
- Quantitative analysis of positional offsets in the dataset
- A prototype repositioning model
- Evaluation results and recommendations for future work

## Skills & Concepts Covered

- Spatial data analysis
- Geospatial error measurement
- Ground-truth dataset construction
- Rapid prototyping and experimentation
- Constrained problem definition in real-world systems

## Project Status

🚧 **Work in progress** — this repository will evolve as experimentation, modeling, and evaluation continue.
