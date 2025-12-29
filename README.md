# MongoLens

![RAG](https://img.shields.io/badge/RAG-grounded%20chat-purple)
![OpenAI](https://img.shields.io/badge/OpenAI-embeddings-yellow)

MongoDB → insights → grounded RAG chat (OpenAI embeddings)

MongoLens is a Streamlit demo app that connects to MongoDB, lists collections, lets you select and export records as JSON, generates basic profiling and chart suggestions, and enables a RAG assistant that answers questions grounded in the exported data with citations.

## Live demo

Add when deployed:

- App: https://mongolens.streamlit.app
- Repo: https://github.com/sc-aisuperjack/mongolens

## Features

- Connect to MongoDB and list collections
- Select one or more collections and sample records
- Export selected collections to JSON (zip)
- Auto profiling and chart suggestions based on the selected data
- RAG chat grounded in retrieved chunks with citations

## Setup

### 1) Create env

Copy `.env` to `.env.local` and fill values:

```bash
cp .env .env.local
```
