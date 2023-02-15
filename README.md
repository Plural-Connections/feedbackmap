# Repo for Feedback Map



To run,
- Ensure your OpenAI API key (from https://platform.openai.com/account/api-keys) is set.  You can set it as the ``OPENAI_API_KEY`` environement variable, or add ``openai.api_key = <API_KEY>`` under ``import openai`` in ``frontend/gpt3_model.py``
- ``pip3 install -r frontend/requirements.txt``
- Within the frontend directory, run ``streamlit run streamlit.py >& streamlit.out &`` to start the streamlit server.

To customize:

- Edit ``frontend/.streamlit/config.toml`` for theme customizations.  (See https://docs.streamlit.io/library/advanced-features/theming for details)
- Edit ``frontend/app_config.py`` to edit the title and some other settings.
