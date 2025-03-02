# AI-Powered Content Extractor

![image](https://github.com/user-attachments/assets/7dbb36d4-8fd7-4500-a794-8cf36aeed420)

## Overview
AI-Powered Content Extractor is a command-line tool that extracts and summarizes content from PDF files and web pages using the power of AI Agents. This tool allows you to process single or multiple sources efficiently and generate concise summaries.

## Features
- Extract content from **PDF files** and **web pages**.
- Summarize the extracted content using AI.
- Support for batch processing of multiple sources.
- Customizable summary styles.
- Configurable AI model and API key.

## Installation
### Prerequisites
- Python 3.9+
- Required dependencies (install using the command below)

```sh
pip install -r requirements.txt
```

## Usage

### Extract and Summarize a Single Source
```sh
python app.py extract "<URL_or_PDF_PATH>" -o <OUTPUT_PATH>
```
Example:
```sh
python app.py extract "https://example.com/article" -o summary.md
```

### Extract and Summarize Multiple Sources
```sh
python app.py batch -f sources.txt -o summaries/
```
Where `sources.txt` contains a list of URLs or file paths (one per line).

### Configuration
You can set up your AI model and API key with the `config` command:
```sh
python app.py config -m <MODEL_ID> -k <API_KEY> -s detailed -l 500
```
Options:
- `-m` : AI model identifier.
- `-k` : API key for the AI service.
- `-s` : Summary style (concise, detailed, bullet-points, executive).
- `-l` : Maximum summary length.

## Demo



https://github.com/user-attachments/assets/f706a271-5393-46a3-ba48-29b2fc8d359d



## License
This project is licensed under the MIT License.

## Contribution
Feel free to contribute! Open an issue or submit a pull request with improvements or feature suggestions.

## Author
[Berkay Güzel](https://medium.com/@berkayguzel43)

Happy extracting! 🚀

