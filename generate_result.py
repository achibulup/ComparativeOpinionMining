import models
import training
import VnCoreNLP
import processing

import json

import os

def generateResult(test_folder: str, result_folder: str, model: models.BertCrfExtractor, vncorenlp: VnCoreNLP.VnCoreNLP):
  print(processing.globDataFiles(test_folder))
  for file in processing.globDataFiles(test_folder):
    with open(file, 'r', encoding="utf-8") as input_file:
      with open(result_folder + os.path.basename(file), 'w', encoding="utf-8") as output_file:
        print("Generating result for " + file)
        for line in input_file.readlines():
          line = line.strip()
          if line == "": continue
          try:
            result = training.predict(model, 
                processing.mineInputSentence(processing.getSentenceFromLine(line), vncorenlp))
            if result[0] != (len(result[1]) != 0):
              print("assertion fail on " + line)
              print(result)
            #   output_file.write(line + "\n\n")
            #   continue
          except Exception as e:
            print("Error while processing " + line)
            print(e)
            output_file.write(line + "\n\n")
            continue
          output_file.write(line + "\n")
          for quintuple in result[1]:
            output_file.write(json.dumps(quintuple, ensure_ascii=False) + "\n")
          output_file.write("\n")
