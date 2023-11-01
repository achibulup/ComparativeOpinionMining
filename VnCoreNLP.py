import subprocess
from typing import TypedDict

class Annotations(TypedDict):
  word_form: list[str] 
  """vd ["Tôi", "là", "sinh_viên"]"""
  pos_tag: list[str]
  """vd ["Nc", "V", "N"]"""
  dep_label: list[str]
  """vd ["nsubj", "cop", "root"]"""
  dep_head: list[int]
  """vd [2, 0, -1]"""
  ner_tag: list[str]
  """vd ["O", "O", "O"]"""

class VnCoreNLP(object):
  def __init__(self, jar_path):
    self._process = subprocess.Popen(
      f'java -Xmx2g -jar {jar_path}'.split(),
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      encoding='utf-8',
    )

  def __enter__(self):
    return self
  
  def __exit__(self, exc_type, exc_value, traceback):
    self._process.stdin.close()
    self._process.stdout.close()
    # wait a bit for the subprocess to exit
    self._process.wait(timeout=1.5)
    if (self._process.poll() is None):
      self._process.kill()

  def annotate(self, text: str) -> list[Annotations]:
    text = text.replace('\r\n', ' ').replace('\n', ' ')
    self._process.stdin.write(text + '\n')
    self._process.stdin.flush()
    sentences = int(self._process.stdout.readline().strip())
    output = []
    for _ in range(sentences):
      annotations = Annotations()
      annotations['word_form'] = []
      annotations['pos_tag'] = []
      annotations['dep_label'] = []
      annotations['dep_head'] = []
      annotations['ner_tag'] = []
      while True:
        line = self._process.stdout.readline()
        if line == '\n' or line == "\r\n":
          break
        tokens = line.split('\t')
        if (len(tokens) != 5):
          raise Exception("Invalid number of tokens: " + line + "")
        annotations['word_form'].append(tokens[0])
        annotations['pos_tag'].append(tokens[1])
        annotations['ner_tag'].append(tokens[2])
        annotations['dep_head'].append(int(tokens[3]) - 1)
        annotations['dep_label'].append(tokens[4])
      output.append(annotations)
    return output