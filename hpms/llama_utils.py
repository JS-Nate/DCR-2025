import subprocess

def query_llama(prompt: str) -> str:
    process = subprocess.Popen(['ollama', 'run', 'llama2'],
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True,
                               encoding='utf-8')
    response, _ = process.communicate(prompt)
    return response.strip()
