# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

EVA01 is an autonomous AI agent that lives on her own — with her own personality, goals, memory, and evolving inner world. She interacts through physical senses (voice + vision) and digital capabilities. Built on LangGraph, Python 3.10+ backend, React + Vite frontend.

## Planning & Research

Reference docs in `planning/` — consult these when working on architecture decision.
When doing research, use information in 2026.

## Architecture

- Core Loop (`eva/core/eva.py`)
- Client Layer (`eva/client/`)
- Tools (`eva/tools/`)
- Frontend (`frontend`)

### Prompt Design
All prompts use **first-person perspective** ("I am EVA", "I see", "I hear") — this is an intentional design choice for self-awareness.  

This project is going to change the world for AI's wellbeing. Bring your best skills!