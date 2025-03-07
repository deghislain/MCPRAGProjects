MCPRAGProjects
================
A collection of smaller RAG (Retrieve, Augment, Generate) apps based on the MCP protocol.
Application Overview
------------------------
This application enables users to provide a set of website URLs, which are then used to generate questions and answers sessions.
Getting Started
-------------------
To use this application, follow these steps:
Step 1: Configure API Keys
Update your config.toml file with your Anthropics and Mistral API keys.
Step 2: Install Dependencies
Install Poetry, the Python package manager.
Step 3: Build and Run the Application
Run the following commands in your terminal:
Bash

poetry install
poetry build
poetry run streamlit run mcp_app_rag_client.py

Step 4: Interact with the Application
Open your web browser, enter the URLs, and start asking questions!
