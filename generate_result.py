import models
import training
import VnCoreNLP
import processing

import os

def generateResult(test_folder: str, result_folder: str, model: models.TheModel, vncorenlp: VnCoreNLP.VnCoreNLP):
  print(processing.globDataFiles(test_folder))
  for file in processing.globDataFiles(test_folder):
    with open(file, 'r', encoding="utf-8") as input_file:
      with open(result_folder + os.path.basename(file), 'w', encoding="utf-8") as output_file:
        print("Generating result for " + file)
        for line in input_file.readlines():
          line = line.strip()
          if line == "": continue
          try:
            result = training.predict(model, processing.getSentenceFromLine(line), vncorenlp)
          except:
            print("Error while processing " + line)
            output_file.write(line + "\n\n")
            continue
          output_file.write(line + "\n")
          if result is not None:
            output_file.write(result + "\n")
          output_file.write("\n")
