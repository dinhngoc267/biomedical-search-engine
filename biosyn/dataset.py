from tqdm import tqdm
import glob
import os
import numpy as np

class QueryDataset():
  def __init__(self, data_dir):
    self.data = self.load_data(data_dir = data_dir)

  def load_data(self, data_dir):
    """
    Parameters
    ----------
    data_dir: a path of data

    Returns
    -------
    data: np.array[(mention, CUI)]
    """
    data = []
    concept_files = glob.glob(os.path.join(data_dir,"*.concept"))
    for concept_file in tqdm(concept_files):
      with open(concept_file, "r", encoding="utf-8") as f:
        concepts = f.readlines()

        for concept in concepts: 
          concept = concept.split("||")
          mention = concept[3].strip()
          cui = concept[4].strip()

          data.append((mention,cui))

    data = np.array(data)

    return data

class DictionaryDataset():
  """
  Dictionary data
  """
  def __init__(self, dictionary_path):
    """
    Parameters
    ----------
    dictionary_path: str
      The path of the dictionary
    """
    self.data = self.load_data(dictionary_path)
  def load_data(self, dictionary_path):
    data = []
    with open(dictionary_path, mode='r', encoding='utf-8') as f:
      lines = f.readlines()
      for line in tqdm(lines):
        line = line.strip()
        if line=="": continue
        cui, name = line.split("||")
        data.append((name,cui))

    data = np.array(data)
    return data
