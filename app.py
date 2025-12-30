import streamlit.web.cli as stcli
import sys

# Wrapper so Hugging Face Spaces (or a top-level `streamlit run`) will run
sys.argv = ["streamlit", "run", "streamlit/app.py", "--server.port", "8501"]

if __name__ == "__main__":
    sys.exit(stcli.main())
