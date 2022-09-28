from torch.utils.data import Dataset
from itertools import islice


class MRPCDataset(Dataset):
     def __init__(self):
        file = open('./data/msr_paraphrase_train.txt', 'r', encoding="utf-8")
        data = []
        for line in islice(file, 1, None):
            content = line.split(sep='\t')
            data.append(content)

        self.data = data

     def __getitem__(self, index):
         if self.data[index][0] == '0':
             label = 0
         else:
             label = 1

         sentences = self.data[index][3] + '[SEP]' + self.data[index][4]
         return sentences, label

     def __len__(self):
         return len(self.data)
