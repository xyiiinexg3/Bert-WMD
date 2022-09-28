from pickletools import optimize
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
from model import FCModel
from dataset import MRPCDataset
from transformers import BertTokenizer, BertModel


class config:
    TRAIN_FILE = "./data/msr_paraphrase_train.txt"

# 1、生成数据迭代器
mrpcDataset = MRPCDataset()
train_loader = DataLoader(dataset=mrpcDataset, batch_size=8, shuffle=True)
print("数据载入")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("配置设备")
# 2、构建模型
# 2.1 分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # bert的基准版本
# 2.2 Bert模型
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.to(device)
print("bert OK")
# 2.3 FC模型
fc_model = FCModel()
fc_model = fc_model.to(device) # 为什么bert不用=
print("FC OK")

# 3、优化器
fc_optimizer = torch.optim.Adam(fc_model.parameters(), lr=0.001)
bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=0.001)

# 4、损失函数
crit = torch.nn.BCELoss()

# 5、计算准确率
def binary_accuracy(predict, label):
    rounded_predict = torch.round(predict) # 四舍五入，相当于阈值为0.5
    correct = (rounded_predict == label).float() # 转化成浮点型，以防除法除尽
    accuracy = correct.sum() / len(correct)
    return accuracy

# 6、训练
def train():
    epoch_loss, epoch_acc = 0., 0.
    total_len = 0

    # 一次迭代
    for i, data in enumerate(train_loader):
        bert_model.train()
        fc_model.train()

        sentence, label = data
        if i == 0:
            print("label.type():", label.type())
            print(label)
        label = label.cuda() # ?

        # 分词编码
        encoding = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        # 模型训练
        bert_output = bert_model(**encoding.to(device)) # 这又是什么 **encoding.to(device)RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument index in method wrapper__index_select)
        pooler_output = bert_output.pooler_output # ？
        predict = fc_model(pooler_output).squeeze()
        # 计算损失、准确率
        loss = crit(predict, label.float()) # 这里需要先float()
        acc = binary_accuracy(predict, label)

        # 更新参数
        fc_optimizer.zero_grad()
        bert_optimizer.zero_grad()
        loss.backward() # 计算梯度
        fc_optimizer.step() # 更新参数
        bert_optimizer.step()

        epoch_loss += loss * len(label) # loss求出的时候取了平均，现在是所有的损失
        epoch_acc += acc * len(label) # 再乘以len就不是准确率了，是准确的数量
        total_len += len(label)

        print("batch %d loss:%f accuracy:%f " % (i, loss, acc))
    return epoch_loss/total_len, epoch_acc/total_len # 算是全部取平均，又变成了准确率


num_epoch = 8
index = 0
for epoch in range(num_epoch):
    epoch_loss, epoch_acc = train()
    print("epoch %d loss:%f accuracy:%f" % (epoch+1, epoch_loss, epoch_acc))


    
