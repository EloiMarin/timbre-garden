# Timbre Garden

Project developed for the [Timbre Tools Hackahton](https://comma.eecs.qmul.ac.uk/timbre-tools-hackathon/).

## The project

We wanted to explore artifical life simulation as a tool for timbrical exploration.
For that, we used [Tölvera](https://github.com/Intelligent-Instruments-Lab/tolvera) and tried to extend it in different ways for timbre exploration:

- Integrate it with [SignalFlow](https://github.com/ideoforms/signalflow)
  - Extend SignalFlow with a node to run [Rave](https://github.com/acids-ircam/RAVE) models
  - Use the [SegmentedGranulator](https://signalflow.dev/library/buffer/granulation/segmentedgranulator/) node combined with feature extraction
  - Use the [Euclidean](https://signalflow.dev/library/sequencing/euclidean/) node to explore the rhythmical dimension
  - Integrate [Taichi](https://www.taichi-lang.org/) into SignalFlow, creating a custom TolveraNode for tight integration of audio data with the life simulation
- Develop a [PureData](https://puredata.info/) patch for fast prototyping of different mappings between the Tölvera simulation and the Rave decoder

The focus of the project was not simply to create a pre-defined sound design for Tölvera, but to provide tools for users of the library to explore their own designs. So our approach was to incorporate first all the tools we wanted to have available for Tölvera, and only then consider the timbre exploration possibilities that they offer.

This turned out to be a way bigger challenge than expected. We quickly found technical challenges in all those exploration lines, which led to most of them to get stuck in this initial technical phase.

So this repository does not contain many demos or examples, simply because we did not have enough time to experiment and create examples, best practices, etc.

## Install and run

WIP
