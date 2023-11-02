def avg(nums: list[float | None]):
  nums = [num for num in nums if num is not None]
  return sum(nums) / len(nums)


class MetricRecord:
  def __init__(self, accuracy = None, precision = None, recall = None, f1 = None):
    self.accuracy: float | None = accuracy
    self.precision: float | None = precision
    self.recall: float | None = recall
    self.f1: float | None = f1

  def __str__(self):
    return f"{{accuracy: {self.accuracy}, precision: {self.precision}, recall: {self.recall}, f1: {self.f1}}}"

class BinaryMetric:
  def __init__(self, *, tp: int = 0, tn: int = 0, fp: int = 0, fn: int = 0):
    self.confusion_matrix = [[fn, fp], [tn, tp]]

  def addSample(self, is_true: bool, is_positive: bool, count = 1):
    self.confusion_matrix[int(is_true)][int(is_positive)] += count

  def numSamples(self):
    return self.confusion_matrix[0][0] + self.confusion_matrix[0][1] + self.confusion_matrix[1][0] + self.confusion_matrix[1][1]

  @property
  def accuracy(self):
    if self.numSamples() == 0:
      return None
    tp = self.confusion_matrix[1][1]
    tn = self.confusion_matrix[1][0]
    return (tp + tn) / self.numSamples()
  
  @property
  def precision(self):
    tp = self.confusion_matrix[1][1]
    fp = self.confusion_matrix[0][1]
    return tp / (tp + fp) if tp + fp != 0 else None
  
  @property
  def recall(self):
    tp = self.confusion_matrix[1][1]
    fn = self.confusion_matrix[0][0]
    return tp / (tp + fn) if tp + fn != 0 else None
  
  @property
  def f1(self):
    precision = self.precision
    recall = self.recall
    if self.precision is None or self.recall is None:
      return None
    if precision + recall == 0:
      return 0
    return 2 * self.precision * self.recall / (self.precision + self.recall)
  
  def asRecord(self):
    return MetricRecord(self.accuracy, self.precision, self.recall, self.f1)

  def __str__(self):
    return str(self.asRecord())

class MultiClassMetric:
  def __init__(self, num_classes: int):
    self._corrects = 0
    self._num_classes = num_classes
    self._tps = [0] * num_classes
    self._actual_counts = [0] * num_classes  
    self._predicted_counts = [0] * num_classes

  @property
  def num_classes(self):
    return self._num_classes

  def addSample(self, actual: int, predicted: int, count = 1):
    self._corrects += count if actual == predicted else 0
    self._tps[actual] += count if actual == predicted else 0
    self._actual_counts[actual] += count
    self._predicted_counts[predicted] += count

  def numSamples(self):
    return sum(self._actual_counts)
  
  @property
  def accuracy(self):
    if self.numSamples() == 0:
      return None
    return self._corrects / self.numSamples()
  
  def getMetric(self, class_index: int):
    tp = self._tps[class_index]
    fn = self._actual_counts[class_index] - tp
    fp = self._predicted_counts[class_index] - tp
    tn = self.numSamples() - tp - fn - fp
    return BinaryMetric(tp=tp, tn=tn, fp=fp, fn=fn)
  
  def __str__(self):
    def displayVal(val: float | None):
      return "{: <12.3f}".format(val) if val is not None else "{: <12}".format("None")
    def displayMetric(name: str, metric: MetricRecord):
      return "{: <8}".format(name) + displayVal(metric.precision) + displayVal(metric.recall) + displayVal(metric.f1)
    metrics = [self.getMetric(i) for i in range(self.num_classes)]
    macro = MetricRecord()
    macro.accuracy = avg([metric.accuracy for metric in metrics])
    macro.precision = avg([metric.precision for metric in metrics])
    macro.recall = avg([metric.recall for metric in metrics])
    macro.f1 = avg([metric.f1 for metric in metrics])
    endl = "\n"
    return f"""[num_samples: {self.numSamples()}, accuracy: {self.accuracy}
        precision   recall      f1
{endl.join([displayMetric(str(i), metrics[i]) for i in range(self.num_classes)])}
{displayMetric("macro", macro)}
]"""
