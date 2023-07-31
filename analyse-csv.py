import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
import argparse


from torch.utils.data import Dataset
from torch import cuda
from tqdm.auto import tqdm
from transformers import BertModel
from transformers import BertTokenizer
from transformers import logging

logging.set_verbosity_error()

LABEL_COLUMNS = ['specific', 'extended', 'categoric', 'omission', 'associate']
BERT_MODEL_NAME = 'bert-base-uncased'
MAX_TOKEN_COUNT = 512
THRESHOLD = 0.55

tokeniser = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

parser = argparse.ArgumentParser(description="AMT")
parser.add_argument('csv')
parser.add_argument('-r', '--response', default='response')  # response column
parser.add_argument('-m', '--model', default='./model.ckpt')

args = parser.parse_args()


class AMSDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokeniser: BertTokenizer,
        max_token_len: int = 128
    ):
        self.tokenizer = tokeniser
        self.data = data
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]

        response = data_row[args.response]

        encoding = self.tokenizer.encode_plus(
            response,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            response=response,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
        )


class AMTModel(pl.LightningModule):
    def __init__(self, bert_model_name, n_classes):
        super().__init__()
        self.bert_model_name = bert_model_name
        self.bert = BertModel.from_pretrained(self.bert_model_name, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output


def get_predictions(model_path, test):
    trained_model = AMTModel.load_from_checkpoint(
        model_path,
        bert_model_name=BERT_MODEL_NAME,
        n_classes=len(LABEL_COLUMNS),
    )
    trained_model.eval()
    trained_model.freeze()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda.empty_cache()
    trained_model = trained_model.to(device)

    test_dataset = AMSDataset(
        test,
        tokeniser,
        max_token_len=MAX_TOKEN_COUNT
    )
    preds = []

    for item in tqdm(test_dataset):
        _, prediction = trained_model(
            item["input_ids"].unsqueeze(dim=0).to(device),
            item["attention_mask"].unsqueeze(dim=0).to(device)
        )
        preds.append(prediction.flatten())

    return preds


data = pd.read_csv(args.csv)
print('Columns in the csv: ', ', '.join(list(data.keys())))
print('Number of responses:', len(data))
print('Analysing the column:', args.response)

predictions = get_predictions(args.model, data)

results = list()
specific = list()
extended = list()
categoric = list()
omission = list()
associate = list()
for row in predictions:
    my_row = list(row)
    i = my_row.index(max(my_row))
    result = 'NONE'
    if my_row[i] >= THRESHOLD:
        result = LABEL_COLUMNS[i]
    results.append(result)
    specific.append(round(float(my_row[0]), 3))
    extended.append(round(float(my_row[1]), 3))
    categoric.append(round(float(my_row[2]), 3))
    omission.append(round(float(my_row[3]), 3))
    associate.append(round(float(my_row[4]), 3))

data['specific'] = specific
data['extended'] = extended
data['categoric'] = categoric
data['omission'] = omission
data['associate'] = associate
data['results'] = results

out = str(args.csv)[:-3]+'results.csv'

data.to_csv(out, index=False)
