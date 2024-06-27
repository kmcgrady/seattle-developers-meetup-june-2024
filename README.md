# Seattle Developer's Meetup Streamlit Demo

From the presentation, "Effortless Interactive UI with Streamlit". This project was written in Python 3.12, but it will likely work with Python 3.8+

To setup:

1. Create an account with [OctoAI](https://octo.ai) and [create an API token](https://octo.ai/docs/getting-started/how-to-create-octoai-access-token).
2. Install Requirements `pip install -r requirements.txt`
3. Set up the secrets for Streamlit.
```
mkdir .streamlit
echo "OCTOAI_API_KEY=\"{PASTE API TOKEN}\"" > .streamlit/secrets.toml
```
4. Run each of the demos with `streamlit run FILE.py`
5. The url will pop up and open a browser window (ususally https://localhost:8501)
6. Cancel the server with Ctrl + C and open a different demo.

Feel free to check out other Streamlit demos with `streamlit hello` or [check out our hosted apps](https://streamlit.io/gallery) on **Streamlit Community Cloud**.

Happy Streamlit-ing!
