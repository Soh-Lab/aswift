# Soh Lab
Custom echem SWV peak extraction code written by Max, data visualization written by Jenny


## Setup
Setup Python 3.12(.7) on your favorite platform and download a C++ compiler

Install Requirements
```
pip install -r requirements.txt
```

## Usage
1. To get started, open example_config.toml and edit fields according to descriptions
2. Run command in terminal to extract peaks and generate json and csv output files:
```
python -m peak_extraction.app -c peak_extraction/config/example_config.toml --save
```
3. Visualize individual fits by running:
```
streamlit run analysis/fit_visualization.py "peak_extraction/config/example_config.toml"
```
To use different config files, you can change the name in the command arguments. This setup is very bare-bones, so happy to implement stuff like more config parameters or a GUI if there is demand.
