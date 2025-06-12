def postprocess(input_txt: str, cleaned_txt: str):
    with open(input_txt, 'r') as f:
        content = f.read().strip()
    with open(cleaned_txt, 'w') as f:
        f.write(content.lower())
